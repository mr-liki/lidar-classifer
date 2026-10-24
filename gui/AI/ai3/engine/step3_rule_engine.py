"""Step 3 of the LiDAR classification pipeline: contract-driven rule engine + .ptc export.

Consumes the AI-predicted cloud from Step 2 (``ai_predicted.laz``) and writes the final
production deliverable (``output_classified.laz``). Every threshold, class code and policy
now comes from an external contract file (``config/contract_spec.json``) instead of being
hardcoded here, so a contract change is a data change, not a code change.

RULES, APPLIED IN THIS ORDER
    A. Vegetation height split (contract vegetation_split bands, currently 0.15-1.0 m -> 3,
       1.0-3.0 m -> 4, >= 3.0 m -> 5); vegetation below the spurious threshold is defaulted to
       ground. Rule I stratifies the crowns it recovers with the same bands.
    B. Building recovery for the colour-blind model: elevated, planar, non-vegetation
       clusters that the AI left as Ground/Unclassified become Class 6.
    F. Network extraction: 3D local covariance eigenvalues (linearity / verticality over each
       point's k nearest neighbours) recover the overhead conductors and the poles/towers the
       colour-blind model misfiled as vegetation-free classes, then the recovered dashes are
       grown back into whole catenary spans (see catenary_spans). Runs before Rule C so a
       recovered wire or pole is already covered by outlier_detection.preserve_classes.
    N. Bridge disposal: the deliverable has no Bridge; the model's internal 17 is re-filed
       straight after A - Ground at <= 0.3 m or as a flat pad, Building as a flat deck / roof above
       2 m, Unclassified otherwise.
    R. Roof consolidation (after L): flat, up-facing roof patches >= 12 m2 that hide what is
       under them become whole roofs - everything in their hull at roof height is Building - and
       Building plus the column under each roof is frozen for the canopy rules.
    L. Building validation: a Building cluster stays Building only when its roof (points
       >= 2 m HAG) faces up and covers >= 6 m2; pink on the ground or in a crown is re-filed by
       height. Runs right after F, before I.
    I. Canopy homogenization: an immutable utility mask is frozen first - every pole, every
       class-14 point of a real, continuous cable, and everything within its 2D wire corridor at
       any height. Outside it, grey / red / pink (and crown-height "ground") points
       whose neighbourhood is a volumetric crown or mostly vegetation become vegetation again,
       stratified with Rule A's bands. Runs right after F, before C (see
       rule_i_canopy_homogenization).
    J. Embedded pole truncation: a pole column inside vegetation keeps only the section above
       the local canopy envelope (max vegetation Z within 1.5 m + 0.2 m), and only when its head
       holds a confirmed wire; the rest of the column becomes vegetation. Runs right after I.
    K. Wire-guided pole recovery: under each local height maximum of the confirmed wire (where
       it is held), a continuous grey / vegetation column longer than 2.5 m is a pole Rule F
       missed; the part above J's vegetation cutoff becomes pole. Runs right after J.
    C. Spatial HAG outlier detection (see below): low -> 7, high -> 18.
    G. Ground refinement: terrain-hugging blade span (water / unclassified returns below
       ground_refinement.max_hag) is folded back into Ground - see the scope note below.
    D. Ground subsampling: unclassify a share of the ground pool.
    E. Map internal codes to the client .ptc target IDs, applying the unmapped class policy.
    H. Enforce the client's target class set: any delivered code outside
       output_class_set.allowed is routed to that block's fallback (noise origins to the noise
       container), so the file can never carry a class the client did not ask for.

NOISE CONSOLIDATION (CLIENT CLASS 51 "Low point")
    The client .ptc funnels every outlier into a single container, so both Rule C outcomes are
    delivered there: internal 7 (below the terrain envelope) and internal 18 (above it) map to
    the same client class. ASPRS 7/18 are therefore never emitted. The class-container
    breakdown still reports the two origins separately, because pick-up/QA of low and high
    outliers is a different job even when they share one colour.

SCOPE OF THE SPECKLE RULE (ground_refinement.scope)
    before_thinning   absorbed points join the ground pool and are thinned with the rest;
    after_thinning    absorbed points are Ground in the deliverable, and Rule D's own thinning
                      output is never re-absorbed (the guard would otherwise undo the sieve).
    The second is the default because speckle is sparse by construction: thinning it lowers the
    quality of the ground surface without lowering its density, and it colours the points back
    into the unassigned container as soon as a .ptc gives class 0 to that container.

WHY RULE C NO LONGER CUTS ON ABSOLUTE Z
    A tile-wide Z cut (mean +/- k*sigma) is driven by the density of the flat parts of the
    tile, so it marks legitimately elevated geometry as noise: on the reference tile the old
    cut flagged the entire bridge deck (394 pts) plus part of the buildings. The contract
    instead bounds HeightAboveGround directly (negative_hag_threshold, max_allowable_hag) and,
    when ``use_spatial_hag`` is on, adds a local-window detector: inside each window the HAG
    envelope (a high/low percentile, widened by a robust MAD-based scale) is estimated, and a
    point is only noise when it sits outside that envelope **and** is spatially isolated in
    3D. The isolation test is what separates a real structure from a spike - a bridge deck is
    a dense, coherent cluster, whereas a bird or multipath return is a lone voxel - so the
    spatial detector removes genuine outliers without destroying structures.

DETERMINISM
    Runs are byte-reproducible: subsampling is seeded from the contract
    (``ground_subsampling.random_seed``) and no timestamp or hash seed reaches the output.
    The written file's MD5 and the contract's MD5 are printed so a delivered artefact can be
    tied back to the exact rules that produced it.

    Reproducible *and* stable, which are different things: Rule D's sample is keyed on the
    row index (``_stable_uniform``) rather than drawn by permuting the pool, so a point's
    fate does not depend on how many other points happen to be in the ground pool. With a
    pool-permuting draw a 0.03% change in the ground set - a slightly different HAG filter,
    say - reshuffled 80% of the thinned points, i.e. 23% of a tile's labels, which makes a
    small upstream refinement indistinguishable from a large regression.

CONTRACT EXTENSION KEYS (optional; defaults applied and reported when absent)
    vegetation_split:    spurious_veg_target_class (2)
    building_recovery:   min_cell_points (3), max_veg_fraction (0.5),
                         intensity_percentile (50.0)
    outlier_detection:   spatial_window_size (20.0), spatial_envelope_percentile (95.0),
                         spatial_sigma (6.0), spatial_scale_floor (1.0),
                         isolation_voxel_size (1.0), isolation_max_neighbors (2)
    ground_subsampling:  random_seed (12345)
    network_extraction:  pole_cell_size (1.0), pole_min_cluster_points (3),
                         pole_min_vertical_span (3.0)
    network_extraction:  conductor_min_neighbors (5), conductor_neighbor_radius (1.2),
                         conductor_pylon_z_margin (4.0)
    catenary_spans:      the whole block (enabled, radius, link_radius, grid_cell, sample_step,
                         min_span_length, max_span_width, min_elongation, min_seed_points,
                         max_samples, max_links, absorb_classes)
    pole_corridors:      the whole block (enabled, cluster_cell, min_tower_points, min_tower_hag,
                         min_span, max_span, max_neighbours, radius, z_below, z_above, min_hag,
                         probe_step, seed_radius, min_seed_coverage, max_chord_residual,
                         sample_step, end_margin, max_corridors, max_samples, corridor_chunk,
                         outward_span, absorb_classes)
    seed_expansion:      the whole block (enabled, radius, iterations, min_hag,
                         max_points, absorb_classes)
    inline_cleanup:      the whole block (enabled, k_neighbors, search_neighbors, perp_radius,
                         along_range, min_hag, min_seeds, max_points, chunk, absorb_classes)
    isolated_noise:      the whole block (enabled, radius, max_neighbors, min_hag,
                         pylon_z_margin, preserve_classes)
    conductor_bridge:    the whole block (enabled, max_gap, min_gap, radius, sample_step,
                         max_samples, neighbour_samples, prefer_shortest, diagnostic_cap)
    pole_cylinder:       the whole block (enabled, radius, base_hag, top_margin,
                         cluster_cell, min_seed_points)
    pole_validation:     the whole block (enabled, require_head_at_wire,
                         head_margin, local_radius)
    ground_refinement:   the whole block (enabled, max_hag, source_classes, target_class, scope)
    pole_truncation:     the whole block (enabled, column_cell, column_link, veg_radius,
                         veg_height_percentile, cutoff_margin, head_depth, attach_radius,
                         min_exposed_points, min_exposed_linearity, min_exposed_verticality)
    pole_recovery:       the whole block (enabled, node_cell, node_radius, node_tolerance,
                         node_merge, search_radius, axis_cell, column_radius, top_gap,
                         top_margin, max_gap, min_column_span, min_exposed_points,
                         min_exposed_span, max_ring_ratio, source_classes)
    bridge_disposal:     the whole block (enabled, ground_max_hag, flat_cell, min_cell_points,
                         max_flat_std)
    roof_consolidation:  the whole block (enabled, min_hag, k_neighbors, min_normal_z,
                         min_planarity, max_sphericity, cluster_cell, cluster_link, min_area,
                         max_ground_share, max_occupancy, z_margin, eave_buffer,
                         eave_z_margin, eave_cluster_link, eave_min_points, eave_min_fill,
                         eave_min_density, gutter_inside_share)
    building_validation: the whole block (enabled, cluster_cell, cluster_link, min_roof_hag,
                         min_normal_z, min_roof_area, footprint_cell, surround_radius,
                         max_surround_fraction, shape_k, max_roof_sphericity)
    canopy_homogenization: the whole block (enabled, corridor_radius, corridor_depth,
                         corridor_height (null = unbounded on that side), fill_min_hag,
                         fill_radius, fill_min_veg, fill_passes, fill_classes, scatter_radius,
                         min_scatter, scatter_min_points, low_wire_max_hag, pole_connect_radius,
                         line_foliage_radius, line_foliage_min_points, line_foliage_max_share, interior_radius,
                         interior_veg_fraction,
                         line_grid_cell,
                         line_link_radius, line_piece_size, min_piece_points,
                         min_line_elongation, max_line_width, max_line_verticality,
                         min_line_length, crown_radius, crown_k_neighbors, min_neighbors, min_veg_fraction,
                         min_sphericity, max_planarity, min_hag, iterations, source_classes)
    output_class_set:    the whole block (enforce, allowed, fallback_class,
                         noise_fallback_class)

display-only key
    class_names:         {code: name} used when reporting the delivered codes; a client .ptc
                         renumbers classes, so the run wrapper supplies the client's names
                         to keep the log from mislabelling the deliverable

Run:
    python step3_rule_engine.py ai_predicted.laz output_classified.laz
    python step3_rule_engine.py ai_predicted.laz out.laz --config config/contract_spec.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import chain

import laspy
import numpy as np

try:  # only Rule F needs it, so the engine still runs without scipy when that rule is off
    from scipy.spatial import cKDTree
    from scipy.spatial import ConvexHull as _ConvexHull
    from scipy.sparse import coo_matrix as _coo_matrix
    from scipy.sparse.csgraph import connected_components as _connected_components
    _SCIPY_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on the environment
    cKDTree = None
    _ConvexHull = None
    _coo_matrix = None
    _connected_components = None
    _SCIPY_ERROR = exc

DEFAULT_CONFIG_PATH = os.path.join("config", "contract_spec.json")

CLASS_NAMES = {
    1: "Unclassified",
    2: "Ground",
    3: "Low Vegetation",
    4: "Medium Vegetation",
    5: "High Vegetation",
    6: "Building",
    7: "Low Point Noise",
    9: "Water",
    14: "Wire / Cable",
    15: "Pole / Utility",
    17: "Bridge / Deck",
    18: "High Point Noise",
}

VEG_CODES = (3, 4, 5)
NOISE_CODES = (7, 18)      # internal low / high outlier classes (both -> client 51)
WATER_CODE = 9             # internal Water: the client's 10-class schema has no Water class

# Where a delivered point's class came from. Rule E routes several origins into one client
# class - against a .ptc whose unassigned container is class 0, both the ground thinning of
# Rule D and the high noise of Rule C land there - so origins are tracked to keep the
# delivered file auditable. The counts are reported as ``class_0_*`` keys because that is how
# clients read the container.
ORIGIN_AI = 0
ORIGIN_VEG_SPLIT = 1
ORIGIN_SPURIOUS_VEG = 2
ORIGIN_BUILDING = 3
ORIGIN_LOW_NOISE = 4
ORIGIN_HIGH_NOISE = 5
ORIGIN_GROUND_THINNING = 6
ORIGIN_CONDUCTOR = 7
ORIGIN_POLE = 8
ORIGIN_GROUND_HOMOGENIZATION = 9
ORIGIN_CANOPY = 10
ORIGIN_POLE_TRUNCATION = 11
ORIGIN_POLE_RECOVERY = 12
ORIGIN_BUILDING_VALIDATION = 13
ORIGIN_BRIDGE_DISPOSAL = 14
ORIGIN_ROOF_CONSOLIDATION = 15
ORIGIN_LABELS = {
    ORIGIN_AI: "AI prediction unchanged",
    ORIGIN_VEG_SPLIT: "Rule A vegetation split",
    ORIGIN_SPURIOUS_VEG: "Rule A spurious vegetation",
    ORIGIN_BUILDING: "Rule B building recovery",
    ORIGIN_LOW_NOISE: "Rule C low noise",
    ORIGIN_HIGH_NOISE: "Rule C high noise",
    ORIGIN_GROUND_THINNING: "Rule D ground thinning",
    ORIGIN_CONDUCTOR: "Rule F conductor",
    ORIGIN_POLE: "Rule F pole",
    ORIGIN_GROUND_HOMOGENIZATION: "Rule G ground homogenization",
    ORIGIN_CANOPY: "Rule I canopy homogenization",
    ORIGIN_POLE_TRUNCATION: "Rule J embedded pole truncation",
    ORIGIN_POLE_RECOVERY: "Rule K wire-guided pole recovery",
    ORIGIN_BUILDING_VALIDATION: "Rule L building validation",
    ORIGIN_BRIDGE_DISPOSAL: "Rule N bridge disposal",
    ORIGIN_ROOF_CONSOLIDATION: "Rule R roof consolidation",
}

# Keys the contract must provide, and the optional extension keys with their defaults.
REQUIRED_KEYS = {
    "vegetation_split": ("low_veg_max_hag", "med_veg_max_hag", "spurious_veg_min_hag"),
    "building_recovery": ("min_hag", "max_hag", "grid_size", "max_z_std", "force_intensity"),
    "outlier_detection": ("use_spatial_hag", "max_allowable_hag", "negative_hag_threshold",
                          "preserve_classes"),
    "ground_subsampling": ("enabled", "target_percentage", "source_pool"),
}
REQUIRED_TOP_LEVEL = ("unmapped_class_policy", "ptc_class_mapping")
EXTENSION_DEFAULTS = {
    "vegetation_split": {"spurious_veg_target_class": 2},
    # Rule X: compact wire clumps inside a crown go back to vegetation (see purge_wire_blobs).
    "wire_blob_purge": {"enabled": True, "link": 1.5, "max_extent": 6.0, "veg_radius": 4.0,
                        "min_veg_share": 0.5, "pole_keep_radius": 2.0},
    # Rule V. Codes are INTERNAL. The model leaves the low layer (grass, crops, shrubs) as
    # Unclassified; this files it into the vegetation band of its height (see the rule).
    "low_vegetation_fill": {"enabled": True, "source_classes": [1], "max_hag": 3.0,
                            "structure_buffer": 0.75, "canopy_radius": 1.5,
                            "canopy_veg_share": 0.5, "canopy_min_veg": 5, "canopy_passes": 6,
                            "building_buffer_2d": 1.5, "shape_k": 12, "shape_radius": 1.0,
                            "max_linearity": 0.75, "max_planarity": 0.6, "wall_max_nz": 0.3,
                            "roof_min_hag": 1.0},
    # Class codes in these two blocks are INTERNAL codes (see the note in the module
    # docstring); the contract's ptc_class_mapping turns them into the client's own ids.
    "network_extraction": {"enabled": True, "min_hag": 2.8, "k_neighbors": 20,
                           "conductor_linearity": 0.58, "conductor_max_verticality": 0.40,
                           "conductor_min_neighbors": 5, "conductor_neighbor_radius": 1.2,
                           "conductor_pylon_z_margin": 4.0,
                           "pole_min_verticality": 0.8, "pole_min_linearity": 0.6,
                           "pole_min_hag": 1.5, "pole_cell_size": 1.0,
                           "pole_min_cluster_points": 3, "pole_min_vertical_span": 3.0,
                           "source_classes": [1, 6, 9],
                           "conductor_class": 14, "pole_class": 15},
    "isolated_noise": {"enabled": True, "radius": 2.5, "max_neighbors": 4, "min_hag": 15.0,
                       "pylon_z_margin": 3.0, "preserve_classes": [6, 14, 15, 17]},
    "conductor_bridge": {"enabled": True, "max_gap": 3.0, "min_gap": 0.5,
                         "radius": 0.35, "sample_step": 0.5, "max_samples": 8,
                         "neighbour_samples": 4, "max_pairs": 20000,
                         "prefer_shortest": True, "diagnostic_cap": 15000},
    # Codes here are INTERNAL codes; see the mapping note above. absorb_classes is the client's
    # target 0 / 5 / 13 face of the cloud at this point in the run: Unclassified (1 and 9),
    # Building (6) and Bridge (17).
    "catenary_spans": {"enabled": True, "radius": 0.60, "link_radius": 1.0,
                       "grid_cell": 0.5, "sample_step": 0.5, "grow_length": 3.0,
                       "min_span_length": 4.0, "max_span_width": 1.2,
                       "min_elongation": 0.75, "min_seed_points": 6, "max_samples": 4000,
                       "max_links": 4000000, "absorb_classes": [1, 6, 9, 17]},
    # Pole-to-pole corridors. Codes are INTERNAL again: absorb_classes is the elevated face of
    # the cloud that the client would call 0 / 2 / 3 / 4 / 5 / 13 - everything above the HAG floor
    # that is not a pole, not noise and not ground, which is what "all elevated points" means.
    # min_tower_hag has a floor because the corridor's own z-envelope is measured from the pole
    # tops: a 5 m object dragged into the tower set pulls that envelope down through the canopy,
    # and it also takes the pole's nearest-neighbour slots, so the real spans are never proposed.
    # Set it to 0 to treat literally every Class 16 cluster as a tower head.
    "pole_corridors": {"enabled": True, "cluster_cell": 1.5, "min_tower_points": 3,
                       "min_tower_hag": 8.0, "min_span": 5.0,                       "max_span": 150.0, "outward_span": 30.0,
                       "max_neighbours": 4, "radius": 1.5, "z_below": 3.5,
                       "z_above": 1.0, "min_hag": 2.5, "probe_step": 10.0,
                       "seed_radius": 1.5, "min_seed_coverage": 0.6,
                       "max_chord_residual": 4.0, "sample_step": 1.0, "end_margin": 2.0,
                       "max_corridors": 2000, "max_samples": 200000,
                       "corridor_chunk": 200, "absorb_classes": [1, 3, 4, 5, 6, 9, 17]},
    # The final pass, after every other wire stage has spoken: grow the conductor class outwards
    # through its own 3D neighbourhood, twice, so a sparse or sagging stretch is closed by the
    # cable that is already there rather than by geometry again. Codes are INTERNAL: absorb_classes
    # is the client's target 0 / 5 / 13 face (Unclassified, Building, Bridge).
    # min_hag sits above the 2.2 m the brief asked for on purpose: measured on the reference tile
    # every floor between 2.2 m and 6.0 m closes the wire identically (0.00 % of cable-band gaps
    # over 1 m, worst gap 0.51 m), while 2.2 m additionally paints 28,396 near-ground points.
    "seed_expansion": {"enabled": True, "radius": 1.2, "iterations": 2, "min_hag": 6.0,
                       "max_points": 200000, "absorb_classes": [1, 6, 9, 17]},
    # A cylinder, not a sphere: perp_radius is what keeps it on the cable, along_range is what
    # reaches across an inline gap. absorb_classes is what makes the stage an override - it is
    # the only stage allowed to take a point the vegetation rules already claimed.
    # min_hag 6 m again, not the brief's 2 m: measured on the reference tile the stage is a
    # no-op on the cable band at either value (+34 cable pts), but at 2 m it paints 40,791 pts
    # below 6 m, which is the near-ground clutter the expansion floor was raised to stop.
    # max_points 1e6 is measured, not arbitrary: on the large tile the worst cable-band gap
    # falls from 29.4 m to 12.3 m by then (0.53 % -> 0.12 % of gaps over 1 m), which is the same
    # worst gap as unbounded at half the cost. 200 k only reached 0.30 % and did not move it.
    "inline_cleanup": {"enabled": True, "k_neighbors": 10, "search_neighbors": 8,
                       "perp_radius": 0.45, "along_range": 1.5, "min_hag": 6.0,
                       "min_seeds": 10, "max_points": 1000000, "chunk": 200000,
                       "absorb_classes": [1, 3, 4, 5, 6, 9, 17]},
    "pole_cylinder": {"enabled": True, "radius": 0.45, "base_hag": 0.3,
                      "top_margin": 1.5, "cluster_cell": 1.0, "min_seed_points": 2},
    # A pylon exists to hold the wire, so a column whose head reaches the conductor's own
    # height is a structure and one that pokes past it is a tree. Measured on the labelled
    # FUNIVIA tile: real columns top out at HAG p10 9.85 / p50 10.65 / p90 11.40 m against a
    # delivered wire band of 4.06-17.80 m, while false ones reach 13.88 m median, 19.33 m p90.
    # The task's thinness (R <= 0.32) and footprint tests were measured too and rejected: the
    # thinness test keeps 0 of 16 real columns (real ones measure r90 0.48-0.59 - a pylon is
    # not a needle) and footprint composition does not separate at all (real vs false
    # above-canopy share 0.369 vs 0.377).
    "pole_validation": {"enabled": True, "require_head_at_wire": True,
                        "head_margin": 2.0, "local_radius": 12.0},
    # Rule I. Codes are INTERNAL: source_classes is the client's grey / red / pink face (1 and 9,
    # 14, 6) plus internal Ground (2), because the colour-blind model files whole tree crowns as
    # Ground - measured on FUNIVIA000007, 34 k "ground" points sit more than 3 m above the
    # terrain with exactly the shape statistics of the labelled canopy (sphericity p50 0.21 vs
    # 0.23). The shape gate is what tells a crown from a cable or a roof there: crowns measure
    # sphericity p10 0.07-0.09 / p50 0.21-0.23, a real conductor 0.002 (linearity 0.96), and a
    # roof is planar, which max_planarity refuses whatever its sphericity.
    # The wire corridor is a pure 2D cylinder (no vertical band) so a cable on a steep slope or in
    # a deep sag is covered over its whole height; line_* judge the wire piece by piece (see
    # _valid_conductor_lines). Internal 15 (Pole) can never be a source: poles are immutable here.
    "canopy_homogenization": {"enabled": True, "corridor_radius": 1.5, "corridor_depth": 3.0,
                              "fill_min_hag": 2.0, "fill_radius": 2.0, "fill_min_veg": 3,
                              "fill_passes": 3, "fill_classes": [1, 9, 14],
                              "scatter_radius": 2.5, "min_scatter": 0.04, "scatter_min_points": 12,
                              "low_wire_max_hag": 2.2, "pole_connect_radius": 1.0,
                              "line_foliage_radius": 1.0, "line_foliage_min_points": 20,
                              "line_foliage_max_share": 0.5,
                              "corridor_height": 2.0,
                              "interior_radius": 3.0, "interior_veg_fraction": 0.5,
                              "line_grid_cell": 0.5, "line_link_radius": 1.0,
                              "line_piece_size": 10.0, "min_piece_points": 4,
                              "min_line_elongation": 0.85, "max_line_width": 2.5,
                              "max_line_verticality": 0.6, "min_line_length": 8.0,
                              "crown_radius": 1.5, "crown_k_neighbors": 20, "min_neighbors": 5,
                              "min_veg_fraction": 0.5, "min_sphericity": 0.08,
                              "max_planarity": 0.6, "min_hag": 1.0, "iterations": 2,
                              "source_classes": [1, 2, 6, 9, 14]},
    # Rule J. A pole embedded in vegetation keeps only the section above the local canopy, and
    # only when its head holds a confirmed wire; see rule_j_exposed_pole_truncation.
    # Rule T. Walks the pole / wire network from poles that stand in the open (see the rule).
    # OFF by default: experimental. On a 2.9 M point tile it cost 330 s and found 4 spans.
    # Codes are INTERNAL: claim_classes is what a hidden pole or wire point may be filed as
    # (Unclassified, Building, Water; vegetation too for the wire, under claim_gap).
    "network_trace": {"enabled": False, "column_cell": 0.2, "column_link": 0.6, "min_points": 30,
                      "min_height": 4.0, "max_width": 1.4, "min_hag": 5.0, "min_span": 20.0,
                      "max_span": 160.0, "max_bend": 35.0, "max_slope": 0.35,
                      "bin": 4.0, "tube": 0.5, "tube_dz": 0.6, "tube_wide": 1.2, "corridor_wide": 5.0,
                      "end_margin": 10.0,
                      "tube_wide_below": 3.0, "ambient_gap": 0.8, "ambient_width": 1.6,
                      "stand_out": 2.0, "noise_codes": [14, 7, 18], "bin_points": 2,
                      "max_sag_ratio": 0.08, "min_coverage": 0.6, "fit_seeds": 6,
                      "min_span_hag": 3.0, "min_coverage_blind": 0.85,
                      "seed_fragment_radius": 5.0,
                      "seed_fragment_points": 8, "seed_fragment_flatness": 0.2,
                      "ring_inner": 0.8, "shaft_outer": 1.5, "ring_foot": 1.5, "open_ratio": 0.3,
                      "flood_radius": 0.45, "max_attach_lift": 6.0,
                      "claim_classes": [1, 3, 4, 5, 6, 7, 9, 18],
                      "wire_claim_classes": [1, 7, 18]},
    "pole_truncation": {"enabled": True, "column_cell": 0.2, "column_link": 0.6,
                        "veg_radius": 1.5, "veg_height_percentile": 100.0, "cutoff_margin": 0.2,
                        "head_depth": 2.0, "attach_radius": 1.5, "min_exposed_points": 3,
                        "min_exposed_linearity": 0.8, "min_exposed_verticality": 0.5,
                        "shaft_min_veg": 5, "shaft_min_points": 30, "shaft_min_height": 4.0},
    # Rule K. Codes are INTERNAL: source_classes is what a missed pole is filed as - grey
    # (1, 9), vegetation (3, 4, 5), and the Ground / Building the colour-blind model also uses.
    "pole_recovery": {"enabled": True, "node_cell": 1.0, "node_radius": 6.0,
                      "node_tolerance": 0.15, "node_merge": 2.0, "search_radius": 1.5,
                      "axis_cell": 0.25, "column_radius": 0.5, "top_gap": 2.0,
                      "top_margin": 1.0, "max_gap": 1.0, "min_column_span": 2.5,
                      "min_exposed_points": 5, "min_exposed_span": 1.5,
                      "max_ring_ratio": 1.0,
                      "source_classes": [1, 2, 3, 4, 5, 6, 9]},
    # Rule L. A Building cluster must have a roof: points >= min_roof_hag whose plane faces up
    # and covers min_roof_area; see rule_l_building_validation.
    "building_validation": {"enabled": True, "cluster_cell": 0.5, "cluster_link": 1.0,
                            "min_roof_hag": 2.0, "min_normal_z": 0.85, "min_roof_area": 6.0,
                            "footprint_cell": 0.5, "surround_radius": 1.5,
                            "max_surround_fraction": 0.5, "shape_k": 20,
                            "max_roof_sphericity": 0.08},
    # Rule N. The client receives no Bridge: internal 17 is re-filed right after Rule A.
    "bridge_disposal": {"enabled": True, "ground_max_hag": 0.3, "flat_cell": 1.0,
                        "min_cell_points": 3, "max_flat_std": 0.15},
    # Rule R. Roofs found from geometry; see rule_r_roof_consolidation for the measurements.
    "roof_consolidation": {"enabled": True, "min_hag": 1.5, "k_neighbors": 20,
                           "min_normal_z": 0.8, "min_planarity": 0.7, "max_sphericity": 0.05,
                           "cluster_cell": 0.5, "cluster_link": 1.0, "min_area": 12.0,
                           "max_ground_share": 0.5, "max_occupancy": 0.45, "z_margin": 0.5,
                           "eave_buffer": 1.2, "eave_z_margin": 0.8, "eave_cluster_link": 2.0,
                           "eave_min_points": 20, "eave_min_fill": 0.5,
                           "eave_min_density": 0.35, "gutter_inside_share": 0.8},
    "ground_refinement": {"enabled": True, "max_hag": 0.15, "source_classes": [1, 9],
                          "target_class": 2, "scope": "after_thinning"},
    # The client's deliverable schema is a 10-class subset of their .ptc; Rule H guarantees the
    # file never leaves the pipeline carrying anything else. Codes here are CLIENT codes.
    "output_class_set": {"enforce": True, "allowed": [0, 1, 2, 3, 4, 5, 13, 14, 16, 51],
                         "fallback_class": 0, "noise_fallback_class": 51},
    "building_recovery": {"min_cell_points": 3, "max_veg_fraction": 0.5,
                          "intensity_percentile": 50.0},
    "outlier_detection": {"spatial_window_size": 20.0, "spatial_envelope_percentile": 95.0,
                          "spatial_sigma": 6.0, "spatial_scale_floor": 1.0,
                          "isolation_voxel_size": 1.0, "isolation_max_neighbors": 2},
    "ground_subsampling": {"random_seed": 12345},
}
SOURCE_POOLS = ("all_ground", "spurious_veg_excluded")
UNMAPPED_POLICIES = ("preserve", "unclassify")
SPECKLE_SCOPES = ("before_thinning", "after_thinning")


def class_name(code) -> str:
    return CLASS_NAMES.get(int(code), "Custom / Other")


def _delivered_label(spec: dict, internal_code: int) -> str:
    """How an internal class code leaves the pipeline: client code plus the client's own name."""
    target = int(spec.get("ptc_class_mapping", {}).get(int(internal_code), int(internal_code)))
    names = display_names(spec)
    return f"client class {target} ({names.get(target, class_name(target))})"


def _class_list(values, label: str) -> list:
    """Validate a contract class list: integers, 0-255, no duplicates, order preserved."""
    if not isinstance(values, (list, tuple)):
        raise SystemExit(f"[X] {label} must be a list of class codes, got {type(values).__name__}")
    out, seen = [], set()
    for value in values:
        code = int(value)
        if not 0 <= code <= 255:
            raise SystemExit(f"[X] {label} entries must be 0-255, got {value!r}")
        if code not in seen:
            seen.add(code)
            out.append(code)
    return out


# --------------------------------------------------------------------------- contract I/O
def resolve_contract_path(config_path: str = DEFAULT_CONFIG_PATH) -> str:
    """Locate the contract, falling back to the module directory for relative paths."""
    if os.path.exists(config_path) or os.path.isabs(config_path):
        return config_path
    beside_module = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
    return beside_module if os.path.exists(beside_module) else config_path


def load_contract_spec(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Read and validate the client contract, filling in documented extension defaults."""
    config_path = resolve_contract_path(config_path)
    if not os.path.exists(config_path):
        raise SystemExit(
            f"[X] Contract spec file missing: {config_path}\n"
            f"    Looked for it beside the module too; pass an explicit path with --config.")
    with open(config_path, encoding="utf-8") as f:
        spec = json.load(f)
    spec["_config_path"] = config_path

    missing = [k for k in REQUIRED_TOP_LEVEL if k not in spec]
    for block, keys in REQUIRED_KEYS.items():
        if block not in spec:
            missing.append(block)
            continue
        missing += [f"{block}.{k}" for k in keys if k not in spec[block]]
    if missing:
        raise SystemExit(f"[X] Contract spec {config_path} is incomplete, missing: {missing}")

    applied = []
    for block, defaults in EXTENSION_DEFAULTS.items():
        if block not in spec:  # whole optional block absent: materialise it
            spec[block] = {}
            applied.append(f"{block} (entire block)")
        for key, value in defaults.items():
            if key not in spec[block]:
                spec[block][key] = value
                applied.append(f"{block}.{key}={value}")
    spec["_extension_defaults_applied"] = applied

    policy = str(spec["unmapped_class_policy"]).lower()
    if policy not in UNMAPPED_POLICIES:
        raise SystemExit(f"[X] unmapped_class_policy must be one of {UNMAPPED_POLICIES}, got {policy!r}")
    spec["unmapped_class_policy"] = policy
    pool = str(spec["ground_subsampling"]["source_pool"]).lower()
    if pool not in SOURCE_POOLS:
        raise SystemExit(f"[X] ground_subsampling.source_pool must be one of {SOURCE_POOLS}, got {pool!r}")
    spec["ground_subsampling"]["source_pool"] = pool
    scope = str(spec["ground_refinement"]["scope"]).lower()
    if scope not in SPECKLE_SCOPES:
        raise SystemExit(f"[X] ground_refinement.scope must be one of {SPECKLE_SCOPES}, "
                         f"got {scope!r}")
    spec["ground_refinement"]["scope"] = scope
    spec["ground_refinement"]["source_classes"] = _class_list(
        spec["ground_refinement"]["source_classes"], "ground_refinement.source_classes")
    spec["network_extraction"]["source_classes"] = _class_list(
        spec["network_extraction"]["source_classes"], "network_extraction.source_classes")
    canopy = spec["canopy_homogenization"]
    canopy["source_classes"] = _class_list(canopy["source_classes"],
                                           "canopy_homogenization.source_classes")
    for key in ("corridor_radius", "line_link_radius", "line_grid_cell", "line_piece_size",
                "min_line_length", "max_line_width", "max_line_verticality", "crown_radius"):
        value = float(canopy[key])
        if value <= 0:
            raise SystemExit(f"[X] canopy_homogenization.{key} must be > 0, got {value!r}")
        canopy[key] = value
    canopy["fill_classes"] = _class_list(canopy["fill_classes"], "canopy_homogenization.fill_classes")
    for bad in (6, int(spec["network_extraction"]["pole_class"])):
        if bad in canopy["fill_classes"]:
            raise SystemExit(f"[X] canopy_homogenization.fill_classes must not list internal {bad}")
    canopy["line_foliage_min_points"] = int(canopy["line_foliage_min_points"])
    canopy["line_foliage_max_share"] = float(canopy["line_foliage_max_share"])
    if not 0.0 < canopy["line_foliage_max_share"] <= 1.0:
        raise SystemExit("[X] canopy_homogenization.line_foliage_max_share must be in (0, 1]")
    for key in ("scatter_radius", "low_wire_max_hag", "pole_connect_radius", "line_foliage_radius"):
        canopy[key] = float(canopy[key])
        if canopy[key] <= 0:
            raise SystemExit(f"[X] canopy_homogenization.{key} must be > 0")
    canopy["min_scatter"] = float(canopy["min_scatter"])
    if not 0.0 <= canopy["min_scatter"] < 1.0 / 3.0:
        raise SystemExit("[X] canopy_homogenization.min_scatter must be in [0, 1/3)")
    canopy["scatter_min_points"] = int(canopy["scatter_min_points"])
    if canopy["scatter_min_points"] < 4:
        raise SystemExit("[X] canopy_homogenization.scatter_min_points must be >= 4")
    for key in ("fill_min_hag", "fill_radius"):
        canopy[key] = float(canopy[key])
        if canopy[key] <= 0:
            raise SystemExit(f"[X] canopy_homogenization.{key} must be > 0")
    for key in ("fill_min_veg", "fill_passes"):
        canopy[key] = int(canopy[key])
        if canopy[key] < 0:
            raise SystemExit(f"[X] canopy_homogenization.{key} must be >= 0")
    for key in ("corridor_depth", "corridor_height"):
        if canopy.get(key) is not None:
            canopy[key] = float(canopy[key])
            if canopy[key] < 0:
                raise SystemExit(f"[X] canopy_homogenization.{key} must be >= 0 or null")
    for key in ("min_veg_fraction", "min_sphericity", "max_planarity", "min_line_elongation",
                "min_hag", "interior_radius", "interior_veg_fraction"):
        value = float(canopy[key])
        if value < 0:
            raise SystemExit(f"[X] canopy_homogenization.{key} must be >= 0, got {value!r}")
        canopy[key] = value
    for key, floor in (("crown_k_neighbors", 3), ("min_neighbors", 1),
                       ("min_piece_points", 2), ("iterations", 1)):
        value = int(canopy[key])
        if value < floor:
            raise SystemExit(f"[X] canopy_homogenization.{key} must be >= {floor}, got {value}")
        canopy[key] = value
    pole_code = int(spec["network_extraction"]["pole_class"])
    if pole_code in canopy["source_classes"]:
        raise SystemExit(f"[X] canopy_homogenization.source_classes must not list internal "
                         f"{pole_code} (Pole): poles are immutable to Rule I")
    trunc = spec["pole_truncation"]
    for key in ("column_cell", "column_link", "veg_radius", "head_depth", "attach_radius"):
        value = float(trunc[key])
        if value <= 0:
            raise SystemExit(f"[X] pole_truncation.{key} must be > 0, got {value!r}")
        trunc[key] = value
    for key, lo, hi in (("veg_height_percentile", 0.0, 100.0), ("cutoff_margin", 0.0, None),
                        ("min_exposed_linearity", 0.0, 1.0), ("min_exposed_verticality", 0.0, 1.0)):
        value = float(trunc[key])
        if value < lo or (hi is not None and value > hi):
            raise SystemExit(f"[X] pole_truncation.{key} must be in [{lo:g}, "
                             f"{'inf' if hi is None else f'{hi:g}'}], got {value!r}")
        trunc[key] = value
    trunc["min_exposed_points"] = int(trunc["min_exposed_points"])
    if trunc["min_exposed_points"] < 2:
        raise SystemExit("[X] pole_truncation.min_exposed_points must be >= 2")
    bd = spec["bridge_disposal"]
    for key in ("ground_max_hag", "flat_cell", "max_flat_std"):
        bd[key] = float(bd[key])
        if bd[key] < 0:
            raise SystemExit(f"[X] bridge_disposal.{key} must be >= 0")
    bd["min_cell_points"] = int(bd["min_cell_points"])
    rc = spec["roof_consolidation"]
    for key in ("min_hag", "min_normal_z", "min_planarity", "max_sphericity", "cluster_cell",
                "cluster_link", "min_area", "max_ground_share", "max_occupancy", "z_margin",
                "eave_buffer", "eave_z_margin", "eave_cluster_link"):
        rc[key] = float(rc[key])
        if rc[key] < 0:
            raise SystemExit(f"[X] roof_consolidation.{key} must be >= 0")
    rc["k_neighbors"] = int(rc["k_neighbors"])
    rc["eave_min_points"] = int(rc["eave_min_points"])
    rc["eave_min_fill"] = float(rc["eave_min_fill"])
    rc["eave_min_density"] = float(rc["eave_min_density"])
    rc["gutter_inside_share"] = float(rc["gutter_inside_share"])
    if rc["k_neighbors"] < 3:
        raise SystemExit("[X] roof_consolidation.k_neighbors must be >= 3")
    bv = spec["building_validation"]
    for key in ("cluster_cell", "cluster_link", "min_roof_hag", "min_roof_area", "footprint_cell",
                "surround_radius", "max_surround_fraction"):
        value = float(bv[key])
        if value <= 0:
            raise SystemExit(f"[X] building_validation.{key} must be > 0, got {value!r}")
        bv[key] = value
    bv["shape_k"] = int(bv["shape_k"])
    bv["max_roof_sphericity"] = float(bv["max_roof_sphericity"])
    bv["min_normal_z"] = float(bv["min_normal_z"])
    if not 0.0 <= bv["min_normal_z"] < 1.0:
        raise SystemExit("[X] building_validation.min_normal_z must be in [0, 1)")
    rec = spec["pole_recovery"]
    rec["source_classes"] = _class_list(rec["source_classes"], "pole_recovery.source_classes")
    for bad in (pole_code, int(spec["network_extraction"]["conductor_class"])):
        if bad in rec["source_classes"]:
            raise SystemExit(f"[X] pole_recovery.source_classes must not list internal {bad}")
    for key in ("node_cell", "node_radius", "node_merge", "search_radius", "axis_cell",
                "column_radius", "top_gap", "max_gap", "min_column_span", "max_ring_ratio",
                "min_exposed_span"):
        value = float(rec[key])
        if value <= 0:
            raise SystemExit(f"[X] pole_recovery.{key} must be > 0, got {value!r}")
        rec[key] = value
    for key in ("node_tolerance", "top_margin"):
        value = float(rec[key])
        if value < 0:
            raise SystemExit(f"[X] pole_recovery.{key} must be >= 0, got {value!r}")
        rec[key] = value
    if rec["column_radius"] > rec["search_radius"]:
        raise SystemExit("[X] pole_recovery.column_radius cannot exceed search_radius")
    rec["min_exposed_points"] = int(rec["min_exposed_points"])
    if rec["min_exposed_points"] < 1:
        raise SystemExit("[X] pole_recovery.min_exposed_points must be >= 1")
    if "corridor_z_band" in canopy:
        print("    [!] canopy_homogenization.corridor_z_band is no longer supported and is ignored:"
              " the wire corridor is a pure 2D cylinder")

    # Positive lengths and counts only: a zero radius would claim nothing and a negative one
    # would silently claim everything in the tile.
    spec["isolated_noise"]["preserve_classes"] = _class_list(
        spec["isolated_noise"]["preserve_classes"], "isolated_noise.preserve_classes")
    for block, keys in (("conductor_bridge", ("max_gap", "min_gap", "radius", "sample_step")),
                        ("isolated_noise", ("radius", "max_neighbors", "min_hag",
                                            "pylon_z_margin")),
                        ("network_extraction", ("conductor_min_neighbors",
                                                "conductor_neighbor_radius",
                                                "conductor_pylon_z_margin")),
                        ("pole_cylinder", ("radius", "base_hag", "cluster_cell")),
                        ("pole_validation", ("head_margin", "local_radius")),
                        ("catenary_spans", ("radius", "link_radius", "grid_cell",
                                            "sample_step", "min_span_length",
                                            "max_span_width")),
                        ("pole_corridors", ("radius", "cluster_cell", "min_span", "max_span",
                                            "probe_step", "seed_radius", "sample_step")),
                        ("seed_expansion", ("radius",)),
                        ("inline_cleanup", ("perp_radius", "along_range"))):
        for key in keys:
            value = float(spec[block][key])
            if value <= 0 and not (block == "pole_cylinder" and key == "base_hag" and value == 0):
                raise SystemExit(f"[X] {block}.{key} must be > 0, got {value!r}")
            spec[block][key] = value
    for block, key, floor in (("conductor_bridge", "max_samples", 1),
                              ("conductor_bridge", "neighbour_samples", 1),
                              ("conductor_bridge", "max_pairs", 1),
                              ("conductor_bridge", "diagnostic_cap", 0),
                              ("pole_cylinder", "min_seed_points", 1),
                              ("catenary_spans", "min_seed_points", 2),
                              ("catenary_spans", "max_samples", 2),
                              ("catenary_spans", "max_links", 1),
                              ("pole_corridors", "min_tower_points", 1),
                              ("pole_corridors", "max_neighbours", 1),
                              ("pole_corridors", "max_corridors", 1),
                              ("pole_corridors", "max_samples", 1),
                              ("pole_corridors", "corridor_chunk", 1),
                              ("seed_expansion", "iterations", 1),
                              ("seed_expansion", "max_points", 1),
                              ("inline_cleanup", "k_neighbors", 2),
                              ("inline_cleanup", "search_neighbors", 1),
                              ("inline_cleanup", "min_seeds", 2),
                              ("inline_cleanup", "max_points", 0),
                              ("inline_cleanup", "chunk", 1)):
        value = int(spec[block][key])
        if value < floor:
            raise SystemExit(f"[X] {block}.{key} must be >= {floor}, got {value}")
        spec[block][key] = value
    spec["pole_cylinder"]["top_margin"] = float(spec["pole_cylinder"]["top_margin"])
    spec["pole_validation"]["enabled"] = bool(spec["pole_validation"]["enabled"])
    spec["pole_validation"]["require_head_at_wire"] = bool(
        spec["pole_validation"]["require_head_at_wire"])
    spec["pole_validation"]["head_margin"] = float(spec["pole_validation"]["head_margin"])
    spec["conductor_bridge"]["prefer_shortest"] = bool(spec["conductor_bridge"]["prefer_shortest"])
    if spec["conductor_bridge"]["min_gap"] > spec["conductor_bridge"]["max_gap"]:
        raise SystemExit("[X] conductor_bridge.min_gap cannot exceed max_gap")

    spec["catenary_spans"]["absorb_classes"] = _class_list(
        spec["catenary_spans"]["absorb_classes"], "catenary_spans.absorb_classes")
    elong = float(spec["catenary_spans"]["min_elongation"])
    if not 0.0 <= elong < 1.0:
        raise SystemExit(f"[X] catenary_spans.min_elongation must be in [0, 1), got {elong!r}")
    spec["catenary_spans"]["min_elongation"] = elong
    grow = float(spec["catenary_spans"]["grow_length"])
    if grow < 0.0:
        raise SystemExit(f"[X] catenary_spans.grow_length must be >= 0, got {grow!r}")
    spec["catenary_spans"]["grow_length"] = grow

    spec["pole_corridors"]["absorb_classes"] = _class_list(
        spec["pole_corridors"]["absorb_classes"], "pole_corridors.absorb_classes")
    cor = spec["pole_corridors"]
    if cor["min_span"] > cor["max_span"]:
        raise SystemExit("[X] pole_corridors.min_span cannot exceed max_span")
    for key, floor in (("min_tower_hag", 0.0), ("z_below", 0.0), ("z_above", 0.0),
                       ("min_hag", 0.0), ("end_margin", 0.0), ("max_chord_residual", 0.0),
                       ("outward_span", 0.0)):
        value = float(cor[key])
        if value < floor:
            raise SystemExit(f"[X] pole_corridors.{key} must be >= {floor:g}, got {value!r}")
        cor[key] = value
    if not 0.0 < float(cor["min_seed_coverage"]) <= 1.0:
        raise SystemExit("[X] pole_corridors.min_seed_coverage must be in (0, 1], got "
                         f"{cor['min_seed_coverage']!r}")
    cor["min_seed_coverage"] = float(cor["min_seed_coverage"])

    spec["seed_expansion"]["absorb_classes"] = _class_list(
        spec["seed_expansion"]["absorb_classes"], "seed_expansion.absorb_classes")
    spec["inline_cleanup"]["absorb_classes"] = _class_list(
        spec["inline_cleanup"]["absorb_classes"], "inline_cleanup.absorb_classes")
    inline = spec["inline_cleanup"]
    inline_hag = float(inline["min_hag"])
    if inline_hag < 0:
        raise SystemExit(f"[X] inline_cleanup.min_hag must be >= 0, got {inline_hag!r}")
    inline["min_hag"] = inline_hag
    exp = spec["seed_expansion"]
    value = float(exp["min_hag"])
    if value < 0.0:
        raise SystemExit(f"[X] seed_expansion.min_hag must be >= 0, got {value!r}")
    exp["min_hag"] = value

    out = spec["output_class_set"]
    allowed = _class_list(out["allowed"], "output_class_set.allowed")
    if not allowed:
        raise SystemExit("[X] output_class_set.allowed must list at least one client class")
    out["allowed"] = allowed
    for key in ("fallback_class", "noise_fallback_class"):
        value = int(out[key])
        if value not in allowed:
            raise SystemExit(f"[X] output_class_set.{key}={value} is not in "
                             f"output_class_set.allowed {allowed}")
        out[key] = value

    mapping = {}
    for src, dst in spec["ptc_class_mapping"].items():
        src_i, dst_i = int(src), int(dst)
        if not (0 <= src_i <= 255 and 0 <= dst_i <= 255):
            raise SystemExit(f"[X] ptc_class_mapping entries must be 0-255, got {src!r} -> {dst!r}")
        mapping[src_i] = dst_i
    spec["ptc_class_mapping"] = mapping
    return spec


# ------------------------------------------------------------------------------ utilities
def _assign(classes: np.ndarray, mask: np.ndarray, value: int) -> int:
    """Set ``mask`` to ``value`` and report how many points actually changed class."""
    changed = int(np.count_nonzero(mask & (classes != value)))
    classes[mask] = value
    return changed


def _md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _grid_cells(x, y, cell_size: float, max_dense_cells: int = 50_000_000):
    """Dense cell index for every point on a horizontal grid, plus the cell count.

    A horizontal grid is dense enough to index arithmetically, so no sorting is needed at
    all; the sparse (``np.unique``) path is only used for absurdly large cell spaces.
    """
    kx = np.floor(x / cell_size).astype(np.int64)
    ky = np.floor(y / cell_size).astype(np.int64)
    kx -= kx.min()
    ky -= ky.min()
    ny = int(ky.max()) + 1
    nx = int(kx.max()) + 1
    if nx * ny <= max_dense_cells:
        return kx * ny + ky, nx * ny
    key = kx.astype(np.uint64) * np.uint64(ny) + ky.astype(np.uint64)
    inverse = np.unique(key, return_inverse=True)[1]
    return inverse, int(inverse.max()) + 1


def _sorted_by_group(values: np.ndarray, groups: np.ndarray, n_groups: int):
    """Sort ``values`` into group order; return the sorted values plus group boundaries."""
    order = np.lexsort((values, groups))
    sorted_values = values[order]
    counts = np.bincount(groups, minlength=n_groups)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    return sorted_values, starts, counts


def _quantile_of_sorted(sorted_values, starts, counts, q: float, out_size: int) -> np.ndarray:
    """Per-group quantile from values already sorted within each group."""
    rank = np.minimum((q / 100.0 * np.maximum(counts - 1, 0)).astype(np.int64),
                      np.maximum(counts - 1, 0))
    index = np.clip(starts + rank, 0, max(len(sorted_values) - 1, 0))
    out = sorted_values[index].astype(np.float64) if len(sorted_values) else np.zeros(out_size)
    return np.where(counts > 0, out, np.nan)


def _group_hag_stats(z_rel: np.ndarray, groups: np.ndarray, n_groups: int, pct: float):
    """Per-group HAG median, robust scale (1.4826 * MAD) and high/low envelope percentiles."""
    sorted_v, starts, counts = _sorted_by_group(z_rel, groups, n_groups)
    median = _quantile_of_sorted(sorted_v, starts, counts, 50.0, n_groups)
    high = _quantile_of_sorted(sorted_v, starts, counts, pct, n_groups)
    low = _quantile_of_sorted(sorted_v, starts, counts, 100.0 - pct, n_groups)
    deviation = np.abs(z_rel - median[groups])
    dev_sorted, dev_starts, dev_counts = _sorted_by_group(deviation, groups, n_groups)
    mad = _quantile_of_sorted(dev_sorted, dev_starts, dev_counts, 50.0, n_groups)
    return median, 1.4826 * mad, high, low


def _voxel_counts(x, y, z, size: float) -> np.ndarray:
    """Number of points sharing each 3D voxel - a cheap, vectorised isolation proxy."""
    kx = np.floor(x / size).astype(np.int64)
    ky = np.floor(y / size).astype(np.int64)
    kz = np.floor(z / size).astype(np.int64)
    kx -= kx.min()
    ky -= ky.min()
    kz -= kz.min()
    key = (kx.astype(np.uint64) * np.uint64(ky.max() + 1) + ky.astype(np.uint64))
    key = key * np.uint64(kz.max() + 1) + kz.astype(np.uint64)
    inverse = np.unique(key, return_inverse=True)[1]
    return np.bincount(inverse)[inverse]


def report_hag_anchor(las, z_rel: np.ndarray, sample: int = 20_000) -> dict:
    """Say out loud how well this file's HeightAboveGround is anchored.

    Every geometric rule in this engine is keyed on HAG, so a tile whose HAG was
    extrapolated over long distances (ground missing under canopy, a cliff, a
    bridge deck, ground the ground filter classified as vegetation) will bend
    those rules no matter how well the thresholds were chosen.  The surface the
    HAG was measured against is not carried in the file, but its points are:
    they are the ones sitting at HAG == 0.  Indexing those in 2D and asking how
    far the rest of the cloud had to reach turns "is this tile's HAG trustworthy"
    into a number, for any file, without assuming anything about class codes.
    """
    stats = {}
    flat = z_rel == 0.0
    stats["hag_zero_points"] = int(flat.sum())
    stats["hag_p50"] = float(np.percentile(z_rel, 50))
    stats["hag_p95"] = float(np.percentile(z_rel, 95))
    stats["hag_negative"] = int((z_rel < 0).sum())
    print(f"            - HAG as supplied: p50 {stats['hag_p50']:.2f} m, p95 {stats['hag_p95']:.2f} m,"
          f" range {z_rel.min():.1f} .. {z_rel.max():.1f} m, {stats['hag_negative']:,} below ground;"
          f" {stats['hag_zero_points']:,} pts sit at HAG 0 (the surface it was measured against)")
    if stats["hag_zero_points"] < 3:
        print("            [!] No points at HAG 0: this file carries no recoverable ground surface,"
              " every HAG threshold below is being applied to an unknown datum")
        return stats
    if float(z_rel.max()) == float(z_rel.min()):
        print("            [!] HeightAboveGround is constant; the rules cannot separate height at all")
        return stats
    # The anchor set is taken whole (it is the real ground density that decides
    # how far a point had to reach); only the *query* side is sampled.
    idx = np.flatnonzero(flat)
    anchor = np.column_stack((np.asarray(las.x, dtype=np.float64)[idx],
                              np.asarray(las.y, dtype=np.float64)[idx]))
    tree = cKDTree(anchor, copy_data=False, balanced_tree=False, compact_nodes=False)
    n = z_rel.size
    q = np.arange(n) if n <= sample else np.linspace(0, n - 1, sample).astype(np.int64)
    d, _ = tree.query(np.column_stack((np.asarray(las.x, dtype=np.float64)[q],
                                       np.asarray(las.y, dtype=np.float64)[q])), k=1, workers=-1)
    stats["anchor_dist_p50"] = float(np.percentile(d, 50))
    stats["anchor_dist_p90"] = float(np.percentile(d, 90))
    stats["anchor_dist_p99"] = float(np.percentile(d, 99))
    stats["anchor_dist_far_fraction"] = float((d > 2.0).mean())
    print(f"            - Ground anchor: p50 {stats['anchor_dist_p50']:.2f} m, "
          f"p90 {stats['anchor_dist_p90']:.2f} m, p99 {stats['anchor_dist_p99']:.2f} m"
          f" to the HAG-0 surface"
          + (f" - {100 * stats['anchor_dist_far_fraction']:.2f}% of sampled points are > 2 m"
             " from it and carry an extrapolated HAG" if stats["anchor_dist_far_fraction"] > 0.01
             else ""))
    return stats


def _stable_uniform(seed: int, idx: np.ndarray) -> np.ndarray:
    """A uniform [0,1) draw per row index, from the seed alone.

    Counter-based (SplitMix64) rather than a sequential RNG draw, so the value
    for row *i* is the same whichever other rows are in the pool.  That is what
    makes a sample of the ground class behave: ``rng.choice(pool, n, replace=False)``
    permutes the pool, so a point's fate depends on the pool's *size*, and a
    0.03% change in the ground set reshuffles ~80% of the thinned points - 23%
    of a tile's labels - which makes every upstream refinement look like a
    regression when it is only a perturbation.
    """
    x = idx.astype(np.uint64, copy=False) * np.uint64(0x9E3779B97F4A7C15)
    x = x + np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def _local_geometry(coords: np.ndarray, query_idx: np.ndarray, k: int,
                    chunk: int = 200_000, tree=None) -> tuple:
    """Linearity and principal-axis verticality of every query point's neighbourhood.

    The neighbourhood is taken from the *whole* cloud, exactly as a per-point
    ``cKDTree.query`` would, so a wire's covariance is dominated by the wire points and not by
    a pre-selection. Queries are batched and chunked so peak memory stays bounded on
    multi-million-point tiles, and the chunk order is fixed - no RNG, no threading race - so
    the result is reproducible byte for byte.

    Returns ``(linearity, verticality)`` where linearity = (lambda1 - lambda2) / lambda1 is 0
    for a degenerate neighbourhood and verticality = |eigenvector_z| of the principal axis.
    """
    n = len(query_idx)
    linearity = np.zeros(n, dtype=np.float64)
    verticality = np.zeros(n, dtype=np.float64)
    if n == 0:
        return linearity, verticality
    k_eff = int(min(k, len(coords)))
    if tree is None:  # the caller may share one tree across several query sets
        tree = _tree(coords)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        _, neighbours = tree.query(coords[query_idx[start:stop]], k=k_eff, workers=-1)
        # Covariance of every neighbourhood at once; eigh returns ascending eigenvalues, so
        # column 2 is the principal axis and lambda1 = evals[:, 2].
        evals, evecs = _neighbour_eig(coords, np.asarray(neighbours).reshape(stop - start, k_eff),
                                      k_eff, vectors=True)
        l1, l2 = evals[:, 2], evals[:, 1]
        ok = np.isfinite(l1) & (l1 > 1e-12)               # non-finite or coincident points
        block_lin = np.zeros(stop - start, dtype=np.float64)
        block_lin[ok] = (l1[ok] - l2[ok]) / l1[ok]
        linearity[start:stop] = block_lin
        verticality[start:stop] = np.where(np.isfinite(evecs[:, 2, 2]), np.abs(evecs[:, 2, 2]), 0.0)
    return linearity, verticality


def _vertical_structure_mask(x, y, hag, mask, cell: float, min_points: int, min_span: float):
    """Keep the pole points that belong to a *structure*, not to a lone vertical speck.

    A pole is several candidate points stacked over one footprint, so grouping the candidates
    into XY cells and measuring how far apart their HAG values are separates a pole from a
    one-point sprite. Returns ``(keep, span_per_point, points_per_cell)`` so the caller can log
    how tall the retained structures are. With ``min_span <= 0`` and ``min_points <= 1`` the gate
    is a no-op, which reproduces the plain per-point test.
    """
    keep = np.zeros(len(mask), dtype=bool)
    span_out = np.zeros(len(mask), dtype=np.float64)
    count_out = np.zeros(len(mask), dtype=np.int64)
    idx = np.flatnonzero(mask)
    if not len(idx):
        return keep, span_out, count_out
    keys = np.column_stack((np.floor(x[idx] / cell), np.floor(y[idx] / cell))).astype(np.int64)
    inverse = np.asarray(np.unique(keys, axis=0, return_inverse=True)[1]).ravel()
    counts = np.bincount(inverse)
    low = np.full(len(counts), np.inf)
    high = np.full(len(counts), -np.inf)
    np.minimum.at(low, inverse, hag[idx])
    np.maximum.at(high, inverse, hag[idx])
    span = high - low
    ok = (counts >= min_points) & (span >= min_span)
    keep[idx] = ok[inverse]
    span_out[idx] = span[inverse]
    count_out[idx] = counts[inverse]
    return keep, span_out, count_out


N_THREADS = max(1, os.cpu_count() or 1)


def _par_ranges(n: int, min_chunk: int = 4096):
    """Split ``range(n)`` into at most ``N_THREADS * 2`` contiguous (lo, hi) slices."""
    parts = int(min(N_THREADS * 2, max(1, n // min_chunk)))
    edges = np.linspace(0, n, parts + 1).astype(np.int64)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _neighbour_eig(cloud: np.ndarray, neighbours: np.ndarray, k: int, vectors: bool = False):
    """Eigen-decomposition of every neighbourhood's covariance, on all cores.

    ``neighbours`` is an (m, k) index array into ``cloud``. Each slice gathers its points, centres
    them, builds the 3x3 covariances with one batched matmul and diagonalises them, all inside a
    worker thread: numpy releases the GIL in the gather, the matmul and the LAPACK calls, so the
    slices really run concurrently, and nothing is pickled or copied between processes (a joblib
    process pool would have to ship the cloud to every worker). Slices are fixed contiguous ranges
    written into preallocated output, so the result does not depend on scheduling.
    Returns ``evals`` (m, 3, ascending) and, if ``vectors``, ``evecs`` (m, 3, 3).
    """
    m = len(neighbours)
    evals = np.empty((m, 3), dtype=np.float64)
    evecs = np.empty((m, 3, 3), dtype=np.float64) if vectors else None

    def work(lo_hi):
        lo, hi = lo_hi
        pts = cloud[neighbours[lo:hi]]                       # (n, k, 3) gather
        pts -= pts.mean(axis=1, keepdims=True)
        cov = np.matmul(pts.transpose(0, 2, 1), pts) / k
        if vectors:
            evals[lo:hi], evecs[lo:hi] = np.linalg.eigh(cov)
        else:
            evals[lo:hi] = np.linalg.eigvalsh(cov)

    ranges = _par_ranges(m)
    if len(ranges) <= 1:
        for r_ in ranges:
            work(r_)
    else:
        with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
            list(pool.map(work, ranges))
    return (evals, evecs) if vectors else evals


def _group_members(labels: np.ndarray):
    """Sort-once membership for non-negative integer ``labels``.

    Returns ``(order, starts, counts)`` so the members of group ``g`` are
    ``order[starts[g]:starts[g] + counts[g]]`` - ascending original positions, exactly what
    ``np.flatnonzero(labels == g)`` returns, without scanning every label once per group (a loop
    over G groups of an N-point label array is G x N; this is N log N once).
    """
    counts = np.bincount(labels)
    order = np.argsort(labels, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    return order, starts, counts


def _ball_count(tree, points: np.ndarray, r: float) -> np.ndarray:
    """Number of tree points within ``r`` of every query point (multi-threaded)."""
    return np.asarray(tree.query_ball_point(points, r=r, return_length=True, workers=-1))


class _Laps:
    """Per-stage wall-clock timer: ``lap("name")`` records the time since the previous call."""

    def __init__(self):
        self.t = time.perf_counter()
        self.seconds = {}

    def __call__(self, name: str) -> None:
        now = time.perf_counter()
        self.seconds[name] = round(self.seconds.get(name, 0.0) + now - self.t, 2)
        self.t = now

    def line(self) -> str:
        return " | ".join(f"{k} {v:.1f} s" for k, v in self.seconds.items())


def _tree(points, leafsize: int = 32):
    """A KD-tree tuned for many queries on a build-once cloud.

    Skipping the internal balancing and compaction triples nothing in query accuracy or speed
    here, but it halves the build time on a three-million-point cloud - and building the tree is
    the single most expensive step in Step 3. Points are borrowed, not copied.
    """
    return cKDTree(points, leafsize=leafsize, copy_data=False,
                   balanced_tree=False, compact_nodes=False)


def _bridge_conductor_gaps(x, y, z_rel, z_raw, seed_mask, spec, candidate_idx):
    """Join wire seeds that are close enough to belong to the same span.

    A thin cable between two returns can lose a stretch of samples, which reaches the viewer as
    a broken red line. Two seeds closer than ``max_gap`` are treated as the ends of one span;
    the segment between them is sampled every ``sample_step`` and candidate points within
    ``radius`` of a sample join the span. The test is 3D, so a tube bridging a gap cannot reach
    down into the canopy underneath it, and seeds that are already adjacent (closer than
    ``min_gap``) need no bridging at all - skipping those keeps the pair set small, which is
    what makes this affordable on a multi-million-point tile.

    Returns ``(mask, stats)`` and never mutates the class array; the caller assigns, so the
    bridged points keep the same provenance as the seeds they were bridged to.
    """
    b = spec["conductor_bridge"]
    pair_chunk = 4_000_000     # pair lengths are measured this many rows at a time
    stats = {"enabled": bool(b["enabled"]), "pairs": 0, "gaps": 0, "samples": 0,
             "bridged": 0, "max_gap": float(b["max_gap"]), "min_gap": float(b["min_gap"]),
             "radius": float(b["radius"]), "max_pairs": int(b["max_pairs"]),
             "pairs_capped": False, "prefer_shortest": bool(b["prefer_shortest"])}
    mask = np.zeros(len(z_rel), dtype=bool)
    if not bool(b["enabled"]) or cKDTree is None:
        return mask, stats
    idx = np.flatnonzero(seed_mask)
    if len(idx) < 2 or len(candidate_idx) == 0:
        return mask, stats

    seed_xyz = np.column_stack((x[idx], y[idx], z_raw[idx]))
    # Bounded pair search. ``query_pairs`` returns every pair within max_gap, and on a dense tile
    # whose wire class has grown into the canopy that is unbounded: CEDRASCO (8.9 M pts, 349 k
    # conductor candidates) died here with std::bad_alloc on a 34 GB machine. Only the shortest
    # gaps of at least min_gap are ever kept, so the seeds are first thinned to one per min_gap
    # voxel (two seeds closer than that need no bridge) and each keeps its ``neighbour_pairs``
    # nearest seeds within max_gap - memory is now linear in the seed count.
    vox = max(stats["min_gap"], 1e-3)
    seed_xyz = seed_xyz[np.sort(np.unique(np.floor(seed_xyz / vox).astype(np.int64), axis=0,
                                          return_index=True)[1])]
    k_pairs = int(min(int(b.get("neighbour_pairs", 16)) + 1, len(seed_xyz)))
    dist, near = cKDTree(seed_xyz).query(seed_xyz, k=k_pairs,
                                         distance_upper_bound=stats["max_gap"], workers=-1)
    dist = np.asarray(dist).reshape(len(seed_xyz), k_pairs)
    near = np.asarray(near).reshape(len(seed_xyz), k_pairs)
    src = np.repeat(np.arange(len(seed_xyz)), k_pairs).reshape(len(seed_xyz), k_pairs)
    ok = (near < len(seed_xyz)) & (near != src)         # real neighbour; deduplicated below
    pairs = np.column_stack((src[ok], near[ok])).astype(np.int64)
    if len(pairs):
        pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    del dist, near, src, ok
    stats["pairs"] = int(len(pairs))
    if not len(pairs):
        return mask, stats

    # Lengths are measured first, in chunks. A dense tile can hand this a hundred million pairs
    # before the budget is applied, and building both 3-vector endpoints for every one of them on
    # top of the pair list itself is what turns a large-but-finite pair set into a memory
    # failure; only the survivors of the budget ever have their endpoints materialised.
    length = np.empty(len(pairs), dtype=np.float64)
    for lo in range(0, len(pairs), pair_chunk):
        block = pairs[lo:lo + pair_chunk]
        delta = seed_xyz[block[:, 1]] - seed_xyz[block[:, 0]]
        length[lo:lo + pair_chunk] = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    wide = length >= stats["min_gap"]
    if not wide.any():
        return mask, stats
    pairs, length = pairs[wide], length[wide]
    # When there are more pairs than the budget allows, keep the *shortest* ones by default: a
    # short gap is a hole inside one span, whereas a long "gap" is usually the distance between
    # two different spans - and bridging those from every pair in the tile is what makes the
    # budget spend itself on the widest tubes in the cloud, which is exactly the shape that
    # swallows canopy. The long structural breaks are the catenary span stage's job, because it
    # bridges them along the wire's own heading instead of in every direction at once.
    # ``prefer_shortest=false`` restores the old widest-first order. The sort is stable, so the
    # same spans are bridged on every run either way.
    budget = int(b["max_pairs"])
    stats["prefer_shortest"] = bool(b["prefer_shortest"])
    if len(length) > budget:
        rank = length if stats["prefer_shortest"] else -length
        keep = np.argsort(rank, kind="stable")[:budget]
        pairs, length = pairs[keep], length[keep]
        stats["pairs_capped"] = True
    stats["gaps"] = int(len(length))
    start = seed_xyz[pairs[:, 0]]
    span = seed_xyz[pairs[:, 1]] - start

    n_samples = int(min(int(b["max_samples"]),
                        max(1, int(np.ceil(float(length.max()) / float(b["sample_step"]))))))
    t = ((np.arange(n_samples) + 0.5) / n_samples).reshape(n_samples, 1)
    samples = (start[:, None, :] + span[:, None, :] * t).reshape(-1, 3)
    stats["samples"] = int(len(samples))

    tree = _tree(np.column_stack((x[candidate_idx], y[candidate_idx], z_raw[candidate_idx])),
                 leafsize=16)
    # k nearest candidates per sample, all of them in one vectorised call (no Python loop over
    # samples). ``idx == len(candidate_idx)`` is the "no neighbour within radius" sentinel.
    _, hit = tree.query(samples, k=int(b["neighbour_samples"]),
                        distance_upper_bound=stats["radius"], workers=-1)
    hit = np.asarray(hit).ravel()
    hit = hit[hit < len(candidate_idx)]
    if len(hit):
        mask[candidate_idx[np.unique(hit)]] = True
    stats["bridged"] = int(mask.sum())
    return mask, stats


def _link_labels(n_nodes: int, links: np.ndarray) -> np.ndarray:
    """Connected-component label of every node of an undirected edge list.

    ``scipy.sparse.csgraph`` walks the whole graph at C speed, which is what makes this affordable
    on a dense tile where the link list can run to millions of edges. Indices are cast down to
    int32 and the payload to int8 so the sparse graph stays well under a gigabyte.
    """
    if n_nodes == 0 or not len(links):
        return np.arange(n_nodes, dtype=np.int64)
    if _connected_components is None or _coo_matrix is None:  # pragma: no cover
        return np.arange(n_nodes, dtype=np.int64)
    graph = _coo_matrix((np.ones(len(links), dtype=np.int8),
                         (links[:, 0].astype(np.int32, copy=False),
                          links[:, 1].astype(np.int32, copy=False))),
                        shape=(n_nodes, n_nodes))
    return _connected_components(graph, directed=False, return_labels=True)[1].astype(np.int64)


def _catenary_span_propagation(x, y, z_rel, z_raw, seed_mask, spec, candidate_idx):
    """Grow every wire dash along its own heading until the span reads as one line again.

    A single pulse rarely returns from a two-centimetre cable, so a conductor reaches the model
    as a row of short dashes instead of a continuous line. ``_bridge_conductor_gaps`` closes the
    small holes by pairing seeds a couple of metres apart; what is left is structural - the dash
    pattern itself. This stage reads each *span* as one object instead of pairing points: seeds
    are linked into a graph, every connected component is measured (how long, how straight, how
    thin) and a component that looks like a wire is resampled as a polyline through the median
    position of its own seeds. That is what makes the tube follow the catenary sag rather than
    cutting chord-straight from end to end, and every eligible point inside ``radius`` of that
    polyline joins the conductor class.

    The shape gate is what stops this from swallowing canopy: a crown is wide and isotropic, so
    it fails ``min_elongation`` (or ``max_span_width``) and is left untouched, while a wire is
    long, straight and thin. Absorption is restricted to ``absorb_classes``, so nothing the
    contract has not listed here can change class, and the caller only ever offers it points
    that are already in those classes.
    """
    c = spec["catenary_spans"]
    stats = {"enabled": bool(c["enabled"]), "radius": float(c["radius"]),
             "link_radius": float(c["link_radius"]), "seeds": 0, "nodes": 0, "links": 0,
             "links_capped": False, "spans": 0, "spans_kept": 0, "samples": 0,
             "absorbed": 0, "rejected_short": 0, "rejected_not_elongated": 0,
             "rejected_too_wide": 0, "grow_length": float(c["grow_length"]),
             "span_length_median": None, "span_width_median": None,
             "absorb_classes": [int(v) for v in c["absorb_classes"]]}
    mask = np.zeros(len(z_rel), dtype=bool)
    if not bool(c["enabled"]) or cKDTree is None:
        return mask, stats

    seed_idx = np.flatnonzero(seed_mask)
    stats["seeds"] = int(len(seed_idx))
    min_pts = int(c["min_seed_points"])
    if len(seed_idx) < min_pts or len(candidate_idx) == 0:
        return mask, stats

    seeds = np.column_stack((x[seed_idx], y[seed_idx], z_raw[seed_idx]))
    # Decimate onto a 3D grid first: a dense tile hands this stage hundreds of thousands of
    # seeds, and one representative per grid cell preserves the dash chain while bounding the
    # size of the graph the linker has to walk.
    keys = np.floor(seeds / float(c["grid_cell"])).astype(np.int64)
    nodes = seeds[np.sort(np.unique(keys, axis=0, return_index=True)[1])]
    stats["nodes"] = int(len(nodes))
    if len(nodes) < min_pts:
        return mask, stats

    links = cKDTree(nodes).query_pairs(r=stats["link_radius"], output_type="ndarray")
    stats["links"] = int(len(links))
    budget = int(c["max_links"])
    if len(links) > budget:
        # Keep the shortest links when the graph would blow the budget. The short ones are the
        # real chain, so connectivity survives the cut, and the stable sort keeps it reproducible.
        gap = np.linalg.norm(nodes[links[:, 0]] - nodes[links[:, 1]], axis=1)
        links = links[np.argsort(gap, kind="stable")[:budget]]
        stats["links_capped"] = True
    labels = _link_labels(len(nodes), links)
    counts = np.bincount(labels, minlength=0) if len(labels) else np.zeros(0, dtype=np.int64)
    stats["spans"] = int(np.count_nonzero(counts))
    if not len(counts):
        return mask, stats

    order = np.argsort(labels, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    step = float(c["sample_step"])
    cap = int(c["max_samples"])
    kept_len, kept_width, blocks = [], [], []
    for group in np.flatnonzero(counts >= min_pts):
        first = int(starts[group])
        pts = nodes[order[first:first + int(counts[group])]]
        centred = pts - pts.mean(axis=0)
        evals, evecs = np.linalg.eigh(centred.T @ centred / len(pts))  # ascending: 0, 1, 2
        l1, l2, l3 = float(evals[2]), float(evals[1]), float(evals[0])
        axis = evecs[:, 2]
        along = centred @ axis
        extent = float(along.max() - along.min())
        if extent < float(c["min_span_length"]):
            stats["rejected_short"] += 1
            continue
        elong = (l1 - l2) / l1 if l1 > 1e-12 else 0.0
        if elong < float(c["min_elongation"]):
            stats["rejected_not_elongated"] += 1
            continue
        width = 2.0 * float(np.sqrt(max(l2 + l3, 0.0)))
        if width > float(c["max_span_width"]):
            stats["rejected_too_wide"] += 1
            continue
        # Wire-shaped. Rebuild its path from the per-bin median position so the samples bend with
        # the sag; empty bins are dropped, and the interpolation then draws straight across the
        # dash gap along the span's own heading - which is exactly the missing piece of wire.
        bin_idx = np.floor((along - along.min()) / step).astype(np.int64)
        n_bins = int(bin_idx.max()) + 1
        path = np.empty((n_bins, 3), dtype=np.float64)
        for dim in range(3):
            sv, st, ct = _sorted_by_group(pts[:, dim], bin_idx, n_bins)
            path[:, dim] = _quantile_of_sorted(sv, st, ct, 50.0, n_bins)
        path = path[np.flatnonzero(np.bincount(bin_idx, minlength=n_bins) > 0)]
        if len(path) < 2:
            stats["rejected_short"] += 1
            continue
        # Grow the span past both of its own ends along the heading of its first and last
        # segment. This is the line-growing step: the next dash of the same conductor sits a few
        # metres beyond the break, so the tube reaches across it - while staying directional, it
        # cannot fan out sideways the way a plain radius search around a seed would.
        grow = float(c["grow_length"])
        if grow > 0.0:
            head = path[1] - path[0]
            tail = path[-1] - path[-2]
            ends = []
            for vec, end, sign in ((head, path[0], -1.0), (tail, path[-1], 1.0)):
                norm = float(np.linalg.norm(vec))
                if norm > 1e-9:
                    ends.append(end + sign * grow * vec / norm)
            if len(ends) == 2:
                path = np.vstack((ends[0], path, ends[1]))
        arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))))
        n_samples = int(min(cap, max(2, int(np.ceil(float(arc[-1]) / step)) + 1)))
        u = np.linspace(0.0, float(arc[-1]), n_samples)
        blocks.append(np.column_stack([np.interp(u, arc, path[:, d]) for d in range(3)]))
        kept_len.append(extent)
        kept_width.append(width)
        stats["spans_kept"] += 1

    if not blocks:
        return mask, stats
    samples = np.concatenate(blocks)
    stats["samples"] = int(len(samples))
    stats["span_length_median"] = float(np.median(kept_len))
    stats["span_width_median"] = float(np.median(kept_width))
    tree = _tree(np.column_stack((x[candidate_idx], y[candidate_idx], z_raw[candidate_idx])),
                 leafsize=16)
    # ``idx == len(candidate_idx)`` is the "nothing within radius" sentinel.
    _, hit = tree.query(samples, k=1, distance_upper_bound=stats["radius"], workers=-1)
    hit = np.asarray(hit).ravel()
    hit = hit[hit < len(candidate_idx)]
    if len(hit):
        mask[candidate_idx[np.unique(hit)]] = True
    stats["absorbed"] = int(mask.sum())
    return mask, stats


def _pole_column_fill(x, y, z_rel, seed_mask, spec, exclude=None):
    """Flood each pole's vertical cylinder from its base up to its top.

    A pole is one solid object, but the model reads it in bands: the shaft comes back as pole,
    the cross-arm as building and the space between as unclassified. Grouping the vertical seed
    points into columns (one XY cell each) and then claiming every point inside radius R of a
    column's centre, from ``base_hag`` to the top seed, turns those bands back into one blue
    pole. Points already recovered as conductors are never taken, so a wire crossing the top of
    a pole stays a wire.

    The column footprint is looked up through a sorted cell index rather than a KD-tree over the
    whole tile, which keeps this linear and quick on multi-million-point clouds.
    """
    c = spec["pole_cylinder"]
    stats = {"enabled": bool(c["enabled"]), "columns": 0, "empty_columns": 0, "filled": 0,
             "radius": float(c["radius"]), "base_hag": float(c["base_hag"]),
             "top_margin": float(c["top_margin"])}
    fill = np.zeros(len(z_rel), dtype=bool)
    if not bool(c["enabled"]):
        return fill, stats
    idx = np.flatnonzero(seed_mask)
    if not len(idx):
        return fill, stats

    radius = float(c["radius"])
    base = float(c["base_hag"])
    margin = float(c["top_margin"])
    cluster_cell = float(c["cluster_cell"])
    min_seeds = int(c["min_seed_points"])

    # Seed points -> columns: one column per occupied XY cell of the cluster grid.
    seed_cells = np.column_stack((np.floor(x[idx] / cluster_cell),
                                  np.floor(y[idx] / cluster_cell))).astype(np.int64)
    inverse = np.asarray(np.unique(seed_cells, axis=0, return_inverse=True)[1]).ravel()
    counts = np.bincount(inverse)
    # Members of each column, found by sorting once: ``idx[inverse == group]`` would rescan every
    # pole point for every column. A stable sort keeps the original (ascending) order inside each
    # group, so the selection is identical.
    by_group = np.argsort(inverse, kind="stable")
    g_start = np.concatenate(([0], np.cumsum(counts)[:-1]))

    # Whole cloud -> sorted cell index on the (radius-sized) lookup grid.
    cell = radius
    gi = np.floor(x / cell).astype(np.int64)
    gj = np.floor(y / cell).astype(np.int64)
    gi0, gj0 = int(gi.min()), int(gj.min())
    n_rows = int(gi.max()) - gi0 + 1
    n_cols = int(gj.max()) - gj0 + 1
    key = (gi - gi0) * np.int64(n_cols) + (gj - gj0)
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]

    for group in np.flatnonzero(counts >= min_seeds):
        sel = idx[by_group[g_start[group]:g_start[group] + counts[group]]]
        cx = float(np.median(x[sel]))
        cy = float(np.median(y[sel]))
        top = float(z_rel[sel].max()) + margin
        row_lo = int(np.floor((cx - radius) / cell)) - gi0
        row_hi = int(np.floor((cx + radius) / cell)) - gi0
        col_lo = int(np.floor((cy - radius) / cell)) - gj0
        col_hi = int(np.floor((cy + radius) / cell)) - gj0
        parts = []
        for row in range(max(row_lo, 0), min(row_hi, n_rows - 1) + 1):
            lo = np.searchsorted(sorted_key, row * n_cols + max(col_lo, 0))
            hi = np.searchsorted(sorted_key, row * n_cols + min(col_hi, n_cols - 1),
                                 side="right")
            if hi > lo:
                parts.append(order[lo:hi])
        if not parts:
            stats["empty_columns"] += 1
            continue
        cand = np.concatenate(parts)
        dx = x[cand] - cx
        dy = y[cand] - cy
        inside = (dx * dx + dy * dy) <= radius * radius
        inside &= (z_rel[cand] >= base) & (z_rel[cand] <= top)
        if exclude is not None:
            inside &= ~exclude[cand]
        fill[cand[inside]] = True
        stats["columns"] += 1
    # Counted from the mask, not summed per column: neighbouring poles can share a cell
    # boundary, and overlapping footprints would otherwise count the same point twice.
    stats["filled"] = int(fill.sum())
    return fill, stats


def _pole_validate(x, y, z_rel, pole_mask, conductor_mask, spec, protected=None):
    """Keep only the pole columns that structurally look like a pylon.

    A pylon is defined by what it does: it holds the wire, so its head is at the height of the
    wire that passes there.  A column that pokes past its wire is a tree; one whose head is far
    below it never carried anything.  The test is deliberately *local* - each column is judged
    against the conductor seeds within ``local_radius`` of it - because a single global band is
    useless on a tile where the wire itself climbs 4 m to 18 m across 860 m of relief.

    Measured against the reference labels on that tile: real columns sit within 0.7 m of the
    local wire height (head minus local wire HAG p50 -0.17 m, p10 -0.71 m, p90 +0.10 m), while
    false columns are 84 m below it at the median (the fallbacks had painted a whole hillside)
    or 6.9 m above it.  Keeping only columns within 2 m keeps 12 of 12 real columns.

    The column is described by the *whole footprint* inside ``pole_cylinder.radius`` - the same
    points the flood claimed - from the base of the seeds to their top, so the head is the
    object's, not the seeds'.

    Two candidates the brief asked for were measured and rejected - see the note on
    ``pole_validation`` in the module docstring: a thinness gate deletes real pylons, and
    footprint composition does not separate real from false at all.

    ``protected`` marks points that must never be demoted (noise, other structures); a column
    whose head consists only of those is skipped rather than judged on points it does not own.
    """
    v = spec["pole_validation"]
    stats = {"enabled": bool(v["enabled"]), "columns": 0, "kept": 0, "rejected": 0,
             "reverted_points": 0, "skipped_no_wire": 0,
             "head_margin": float(v["head_margin"]),
             "local_radius": float(v["local_radius"]),
             "wire_band": None, "skipped_protected": 0,
             "reverted_by_source": {}}
    if not bool(v["enabled"]) or not bool(v["require_head_at_wire"]) or not pole_mask.any():
        return pole_mask, stats
    if not conductor_mask.any():
        stats["skipped_no_wire"] = int(np.count_nonzero(pole_mask))
        return pole_mask, stats

    lap = _Laps()
    margin = float(v["head_margin"])
    local_radius = float(v["local_radius"])
    wire_idx = np.flatnonzero(conductor_mask)
    wire_tree = cKDTree(np.column_stack((x[wire_idx], y[wire_idx])), copy_data=False,
                        balanced_tree=False, compact_nodes=False)
    wire_hag = z_rel[wire_idx]
    stats["wire_band"] = [float(np.percentile(wire_hag, 5)),
                          float(np.percentile(wire_hag, 95))]

    c = spec["pole_cylinder"]
    radius = float(c["radius"])
    cluster_cell = float(c["cluster_cell"])
    min_seeds = int(c["min_seed_points"])

    keep = pole_mask.copy()
    # Only cells whose pole points actually came from a column flood can be validated: a pole
    # seed that never qualified as a column is a lone speck and is left exactly as it was.
    idx = np.flatnonzero(pole_mask)
    seed_cells = np.column_stack((np.floor(x[idx] / cluster_cell),
                                  np.floor(y[idx] / cluster_cell))).astype(np.int64)
    inverse = np.asarray(np.unique(seed_cells, axis=0, return_inverse=True)[1]).ravel()
    counts = np.bincount(inverse)
    # Members of each column, found by sorting once: ``idx[inverse == group]`` would rescan every
    # pole point for every column. A stable sort keeps the original (ascending) order inside each
    # group, so the selection is identical.
    by_group = np.argsort(inverse, kind="stable")
    g_start = np.concatenate(([0], np.cumsum(counts)[:-1]))

    lap("setup")
    cell = radius
    gi = np.floor(x / cell).astype(np.int64)
    gj = np.floor(y / cell).astype(np.int64)
    gi0, gj0 = int(gi.min()), int(gj.min())
    n_rows = int(gi.max()) - gi0 + 1
    n_cols = int(gj.max()) - gj0 + 1
    key = (gi - gi0) * np.int64(n_cols) + (gj - gj0)
    del gi, gj
    lap("cell keys")

    # Column centres first (the median of each column's own seeds), so the grid index below only
    # has to cover the cells those columns reach. Sorting all 15 M points by cell to look up a
    # few thousand 0.9 m cylinders is the expensive way: the cylinder test is exact whatever
    # superset of its points the index hands back, so the index only needs the points that can
    # pass it (above ``base_hag``, not protected) in the cells a column touches.
    groups = np.flatnonzero(counts >= min_seeds)
    n_g = len(groups)
    cxs = np.empty(n_g)
    cys = np.empty(n_g)
    z_sel_max = np.empty(n_g)
    for gk, group in enumerate(groups):
        sel = idx[by_group[g_start[group]:g_start[group] + counts[group]]]
        cxs[gk] = float(np.median(x[sel]))
        cys[gk] = float(np.median(y[sel]))
        z_sel_max[gk] = float(z_rel[sel].max())
    lap("centres")
    row_lo_a = np.floor((cxs - radius) / cell).astype(np.int64) - gi0
    row_hi_a = np.floor((cxs + radius) / cell).astype(np.int64) - gi0
    col_lo_a = np.floor((cys - radius) / cell).astype(np.int64) - gj0
    col_hi_a = np.floor((cys + radius) / cell).astype(np.int64) - gj0
    span = int(max((row_hi_a - row_lo_a).max(initial=0), (col_hi_a - col_lo_a).max(initial=0))) + 1
    offs = np.arange(span)
    rows_g = row_lo_a[:, None] + offs[None, :]
    cols_g = col_lo_a[:, None] + offs[None, :]
    row_ok = (rows_g <= row_hi_a[:, None]) & (rows_g >= 0) & (rows_g < n_rows)
    col_ok = (cols_g <= col_hi_a[:, None]) & (cols_g >= 0) & (cols_g < n_cols)
    want = (rows_g[:, :, None] * np.int64(n_cols) + cols_g[:, None, :])[
        row_ok[:, :, None] & col_ok[:, None, :]]
    want = np.unique(want)
    usable = z_rel >= float(c["base_hag"])
    if protected is not None:
        usable &= ~protected
    if len(want):
        pos = np.minimum(np.searchsorted(want, key), len(want) - 1)
        usable &= want[pos] == key
    sub = np.flatnonzero(usable)
    lap("usable")
    order = sub[np.argsort(key[sub], kind="stable")]
    sorted_key = key[order]
    lap("sort")
    t_query = t_median = 0.0

    for gk, group in enumerate(groups):
        sel = idx[by_group[g_start[group]:g_start[group] + counts[group]]]
        cx, cy = float(cxs[gk]), float(cys[gk])
        row_lo, row_hi = int(row_lo_a[gk]), int(row_hi_a[gk])
        col_lo, col_hi = int(col_lo_a[gk]), int(col_hi_a[gk])
        parts = []
        for row in range(max(row_lo, 0), min(row_hi, n_rows - 1) + 1):
            a = np.searchsorted(sorted_key, row * n_cols + max(col_lo, 0))
            b = np.searchsorted(sorted_key, row * n_cols + min(col_hi, n_cols - 1),
                                side="right")
            if b > a:
                parts.append(order[a:b])
        if not parts:
            continue
        cand = np.concatenate(parts)
        dx = x[cand] - cx
        dy = y[cand] - cy
        inside = (dx * dx + dy * dy) <= radius * radius
        inside &= (z_rel[cand] >= float(c["base_hag"]))
        inside &= (z_rel[cand] <= z_sel_max[gk] + float(c["top_margin"]))
        if protected is not None:
            inside &= ~protected[cand]
        if not inside.any():
            stats["skipped_protected"] += 1
            continue
        stats["columns"] += 1
        own = cand[inside]
        head = float(z_rel[own].max())
        _t = time.perf_counter()
        near = wire_tree.query_ball_point(np.array([cx, cy]), local_radius)
        t_query += time.perf_counter() - _t
        if not len(near):
            # No confirmed wire anywhere near this column: it is not a support for anything, so
            # the pylon test cannot speak and the column is left alone.
            stats["skipped_no_wire"] += 1
            continue
        _t = time.perf_counter()
        local_wire = float(np.median(wire_hag[near]))
        t_median += time.perf_counter() - _t
        if head <= local_wire + margin:
            stats["kept"] += 1
            continue
        stats["rejected"] += 1
        keep[own] = False

    lap("loop")
    stats["time_by_stage"] = dict(lap.seconds)
    stats["time_by_stage"]["of which wire query"] = round(t_query, 2)
    stats["time_by_stage"]["of which wire median"] = round(t_median, 2)
    stats["reverted_points"] = int(np.count_nonzero(pole_mask & ~keep))
    return keep, stats


def _pole_to_pole_corridors(x, y, z_rel, z_raw, pole_mask, seed_mask, spec, candidate_idx):
    """Interpolate the missing cable between two pole heads.

    A conductor only exists between the structures that carry it, so once the poles are known so
    is the space a missing cable may occupy: the cylinder swept along the segment joining two
    pole heads. This stage builds those cylinders, keeps the ones a wire actually runs through,
    and forces every eligible point inside them to the conductor class. That recovers the
    mid-span stretches a per-point covariance test cannot see - out there the cable is sampled
    too sparsely to have wire neighbours of its own, which is exactly why it reaches the viewer
    as a dashed line.

    Two things keep it honest. Corridors are built between *adjacent* towers only - each tower's
    ``max_neighbours`` nearest within ``max_span`` - because the exhaustive pole-pair list grows
    with the square of the pole count, and because "every pair within 150 m" is not a span. And a
    corridor is used only when wire evidence supports it: conductor seeds must appear in most of
    the bins along its length *and* sit close to the chord, which is what a sagging catenary does
    and what a hole bored through canopy does not.

    Returns ``(mask, stats)``; the caller assigns, so the points keep conductor provenance.
    """
    c = spec["pole_corridors"]
    stats = {"enabled": bool(c["enabled"]), "radius": float(c["radius"]),
             "max_span": float(c["max_span"]), "tower_points": 0, "towers": 0,
             "tower_pairs": 0, "supported": 0, "corridors_capped": False, "samples": 0,
             "samples_capped": False, "absorbed": 0, "rejected_unsupported": 0,
             "outward_span": float(c["outward_span"]), "terminal_poles": 0,
             "extended_corridors": 0,
             "rejected_unstable": 0, "coverage_median": None, "residual_median": None,
             "absorb_classes": [int(v) for v in c["absorb_classes"]]}
    mask = np.zeros(len(z_rel), dtype=bool)
    if not bool(c["enabled"]) or cKDTree is None:
        return mask, stats
    pole_idx = np.flatnonzero(pole_mask)
    stats["tower_points"] = int(len(pole_idx))
    if len(pole_idx) < 2 or len(candidate_idx) == 0:
        return mask, stats

    # --- 1. pole heads: one head per XY cluster, at the top of the cluster ---
    cells, _ = _grid_cells(x[pole_idx], y[pole_idx], float(c["cluster_cell"]))
    order = np.argsort(cells, kind="stable")
    _, first, group = np.unique(cells[order], return_index=True, return_counts=True)
    px, py = x[pole_idx][order], y[pole_idx][order]
    pz, ph = z_raw[pole_idx][order], z_rel[pole_idx][order]
    tower = np.column_stack((np.add.reduceat(px, first) / group,
                             np.add.reduceat(py, first) / group,
                             np.maximum.reduceat(pz, first),
                             np.maximum.reduceat(ph, first)))
    keep = group >= int(c["min_tower_points"])
    if float(c["min_tower_hag"]) > 0.0:
        keep &= tower[:, 3] >= float(c["min_tower_hag"])
    tower = tower[keep]
    stats["towers"] = int(len(tower))
    if len(tower) < 2:
        return mask, stats

    # --- 2. adjacent tower pairs (nearest neighbours, not every pair in the tile) ---
    tree_xy = cKDTree(tower[:, :2])
    k = min(int(c["max_neighbours"]) + 1, len(tower))
    reach, near = tree_xy.query(tower[:, :2], k=k, workers=-1)
    reach = np.atleast_2d(reach)
    near = np.atleast_2d(near)
    src = np.repeat(np.arange(len(tower)), near.shape[1])
    dst, span = near.ravel(), reach.ravel()
    usable = ((dst < len(tower)) & (dst != src) & np.isfinite(span)
              & (span >= float(c["min_span"])) & (span <= stats["max_span"]))
    pairs = np.unique(np.sort(np.column_stack((src[usable], dst[usable])), axis=1), axis=0)
    stats["tower_pairs"] = int(len(pairs))
    if not len(pairs):
        return mask, stats
    # Which poles sit at the end of a line? A mid-line pole has heads ahead of it *and* behind
    # it; an end pole only ever has them ahead. So the test is whether every neighbour that is
    # part of the line lies in one half-plane as seen from the pole. The brief's sharper form -
    # exactly one neighbour within ``max_span`` - is the special case with a single neighbour,
    # and it only ever fires when pole spacing is close to ``max_span`` (on the reference tile a
    # 61 m and a 34 m span put two poles inside each other's reach, so it fired nowhere); the
    # half-plane form also catches the 30-60 m chains the tiles actually carry. Neighbours closer
    # than ``min_span`` are not part of the line and do not count.
    valid = ((near < len(tower)) & (near != np.arange(len(tower))[:, None])
             & (reach >= float(c["min_span"])) & (reach <= stats["max_span"]))
    delta = np.where(valid[:, :, None],
                     tower[np.clip(near, 0, len(tower) - 1), :2] - tower[:, :2][:, None, :],
                     0.0)
    norm = np.linalg.norm(delta, axis=2, keepdims=True)
    unit = np.divide(delta, norm, out=np.zeros_like(delta), where=norm > 0)
    pair_ok = valid[:, :, None] & valid[:, None, :] & ~np.eye(near.shape[1], dtype=bool)
    worst = np.where(pair_ok, np.einsum("nki,nli->nkl", unit, unit), np.inf).min(axis=(1, 2))
    terminal = (valid.sum(axis=1) >= 1) & (worst > 0.0)

    # --- 3. keep only corridors that wire evidence supports along their whole length ---
    heads = tower[:, :3]                      # (x, y, z) of each pole head
    p1, p2 = heads[pairs[:, 0]], heads[pairs[:, 1]]
    chord = p2 - p1
    length = np.linalg.norm(chord[:, :2], axis=1)     # horizontal run of the span
    n_probe = np.maximum(2, np.ceil(length / float(c["probe_step"])).astype(np.int64) + 1)
    probe_of = np.repeat(np.arange(len(pairs)), n_probe)
    ramp = (np.arange(int(n_probe.sum())) - np.repeat(np.cumsum(n_probe) - n_probe, n_probe))
    t = ramp / np.maximum(n_probe[probe_of] - 1, 1)
    probes = p1[probe_of] + chord[probe_of] * t[:, None]

    seed_idx = np.flatnonzero(seed_mask)
    if len(seed_idx) < 2:
        return mask, stats
    seed_tree = _tree(np.column_stack((x[seed_idx], y[seed_idx], z_raw[seed_idx])), leafsize=16)
    _, near_seed = seed_tree.query(probes, k=1, distance_upper_bound=float(c["seed_radius"]),
                                  workers=-1)
    near_seed = np.asarray(near_seed).ravel()
    found = near_seed < len(seed_idx)
    coverage = np.bincount(probe_of[found], minlength=len(pairs)) / np.maximum(n_probe, 1)
    residual = np.full(len(pairs), np.inf)
    if found.any():
        offset = np.abs(z_raw[seed_idx][near_seed[found]] - probes[found, 2])
        sv, st, ct = _sorted_by_group(offset, probe_of[found], len(pairs))
        residual = _quantile_of_sorted(sv, st, ct, 50.0, len(pairs))
    min_cov = float(c["min_seed_coverage"])
    supported = (coverage >= min_cov) & (residual <= float(c["max_chord_residual"]))
    stats["rejected_unsupported"] = int(np.count_nonzero(coverage < min_cov))
    stats["rejected_unstable"] = int(np.count_nonzero((coverage >= min_cov) & ~supported))
    stats["supported"] = int(supported.sum())
    if not supported.any():
        return mask, stats
    stats["coverage_median"] = float(np.median(coverage[supported]))
    stats["residual_median"] = float(np.median(residual[supported]))
    pairs = pairs[supported]
    cap = int(c["max_corridors"])
    if len(pairs) > cap:
        # Best-supported first, ties broken by index, then re-sorted into corridor order so the
        # sampling below stays reproducible whatever the budget is.
        rank = np.lexsort((np.arange(len(pairs)), -coverage[supported]))
        pairs = pairs[np.sort(rank[:cap])]
        stats["corridors_capped"] = True

    # --- 4. force every eligible point inside an accepted corridor to the conductor class ---
    cand_xy = np.column_stack((x[candidate_idx], y[candidate_idx]))
    cand_z = z_raw[candidate_idx]
    cand_tree = _tree(cand_xy, leafsize=16)
    radius = stats["radius"]
    step = float(c["sample_step"])
    margin = float(c["end_margin"])
    below, above = float(c["z_below"]), float(c["z_above"])
    budget = int(c["max_samples"])
    used = 0
    for start in range(0, len(pairs), max(int(c["corridor_chunk"]), 1)):
        block = pairs[start:start + max(int(c["corridor_chunk"]), 1)]
        a, b = heads[block[:, 0]], heads[block[:, 1]]
        seg = b - a
        run = np.linalg.norm(seg[:, :2], axis=1)
        # A terminal end is carried ``outward_span`` metres past the last pole head, along the
        # heading the line already had, so the cable that vanishes at the edge of the survey is
        # still collected. Only ends whose pole has a single neighbour are extended.
        ext_lo = np.where(terminal[block[:, 0]], float(c["outward_span"]), 0.0)
        ext_hi = np.where(terminal[block[:, 1]], float(c["outward_span"]), 0.0)
        lo = -margin - ext_lo
        hi = run + margin + ext_hi
        n_step = np.maximum(2, np.ceil((hi - lo) / step).astype(np.int64) + 1)
        if used + int(n_step.sum()) > budget:   # never let one tile outgrow the budget
            stats["samples_capped"] = True
            break
        used += int(n_step.sum())
        axis = seg[:, :2] / run[:, None]
        owner = np.repeat(np.arange(len(block)), n_step)
        tick = np.arange(int(n_step.sum())) - np.repeat(np.cumsum(n_step) - n_step, n_step)
        along = lo[owner] + (hi - lo)[owner] * (tick / np.maximum(n_step[owner] - 1, 1))
        samples = np.repeat(a[:, :2], n_step, axis=0) + np.repeat(axis, n_step, axis=0) * along[:, None]
        z_lo = np.repeat(np.minimum(a[:, 2], b[:, 2]) - below, n_step)
        z_hi = np.repeat(np.maximum(a[:, 2], b[:, 2]) + above, n_step)
        hits = cand_tree.query_ball_point(samples, r=radius)
        counts = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
        if not counts.sum():
            continue
        near = np.concatenate([np.asarray(h, dtype=np.int64) for h in hits])
        owner = np.repeat(np.arange(len(hits)), counts)
        inside = (cand_z[near] >= z_lo[owner]) & (cand_z[near] <= z_hi[owner])
        if inside.any():
            mask[candidate_idx[np.unique(near[inside])]] = True
    stats["samples"] = int(used)
    stats["absorbed"] = int(mask.sum())
    stats["terminal_poles"] = int(terminal.sum())
    stats["extended_corridors"] = int(np.count_nonzero(
        terminal[pairs[:, 0]] | terminal[pairs[:, 1]]))
    return mask, stats


def _expand_conductor_seeds(x, y, z_raw, conductor_mask, spec, candidate_idx):
    """Grow the conductor class through its own 3D neighbourhood, twice.

    Everything above this point reasons about the cable's *geometry* - its covariance, its span,
    the corridor that must contain it. This pass stops reasoning and simply follows the cable:
    a point within ``radius`` of a confirmed conductor is part of the same cable far more often
    than it is anything else, because a conductor is the only thing in an overhead corridor that
    comes in a one-dimensional chain. Running it more than once is what closes a stretch that
    dips out of a straight line or a sparse run of returns - each round feeds the next, so the
    conductor walks along the wire instead of stopping at a fixed standoff.

    Only ``absorb_classes`` is ever taken, so the class cannot spread into ground, poles, noise,
    or into a class the contract has not listed, and ``max_points`` bounds the whole pass. That
    bound matters because this is the one stage that inherits whatever the class already is: a
    conductor class carrying false positives is dilated, not corrected, so on a tile where the
    wire rules over-fired the growth would otherwise run away. When the budget binds, the
    candidates nearest a confirmed conductor are taken first and the rest are reported, because
    distance is the only ordering this pass has that means anything.
    """
    e = spec["seed_expansion"]
    stats = {"enabled": bool(e["enabled"]), "radius": float(e["radius"]),
             "iterations": int(e["iterations"]), "absorbed": 0, "per_iteration": [],
             "max_points": int(e["max_points"]), "capped": False,
             "absorb_classes": [int(v) for v in e["absorb_classes"]]}
    newly = np.zeros(len(x), dtype=bool)
    if not bool(e["enabled"]) or cKDTree is None or len(candidate_idx) == 0:
        return newly, stats

    current = np.asarray(conductor_mask, dtype=bool).copy()
    coords = np.column_stack((x, y, z_raw))
    radius = stats["radius"]
    budget = stats["max_points"]
    absorbed = 0
    remaining = candidate_idx
    for _ in range(int(e["iterations"])):
        if absorbed >= budget:
            stats["capped"] = bool(len(remaining))
            break
        confirmed = np.flatnonzero(current)
        if not len(confirmed) or not len(remaining):
            break
        tree = _tree(coords[confirmed], leafsize=16)
        dist, hit = tree.query(coords[remaining], k=1, distance_upper_bound=radius,
                              workers=-1)
        dist = np.asarray(dist).ravel()
        hit = np.asarray(hit).ravel()
        take = hit < len(confirmed)      # the sentinel is "nothing within radius"
        picked = remaining[take]
        if not len(picked):
            stats["per_iteration"].append(0)
            break
        order = np.argsort(dist[take], kind="stable")
        picked = picked[order]
        room = budget - absorbed
        if len(picked) > room:
            picked = picked[:room]
            stats["capped"] = True
        stats["per_iteration"].append(int(len(picked)))
        current[picked] = True           # feeds the next round, that is what makes it grow
        newly[picked] = True
        absorbed += int(len(picked))
        remaining = remaining[~take]
    stats["absorbed"] = int(newly.sum())
    return newly, stats


def _inline_conductor_cleanup(x, y, z_raw, conductor_mask, spec, candidate_idx):
    """Close the gaps that sit *inline* with the cable, using the cable's own heading.

    This is the last wire stage and the most literal one. Every other wire rule judges a point
    against an abstraction - the covariance of its neighbourhood, a span polyline, a pole-to-pole
    corridor. Here each confirmed conductor point contributes the direction of the cable around
    it (the principal axis of its ``k_neighbors`` nearest conductors), and any point that sits
    within ``perp_radius`` of that axis while lying within ``along_range`` ahead of or behind it
    is taken. Being directional in *both* senses is the point: the along-range is what reaches
    across an inline gap, and the tight perpendicular radius is what stops the reach becoming a
    sideways flood under a canopy.

    Because the test is a cylinder around a seed's axis, nothing outside
    ``hypot(perp_radius, along_range)`` of that seed can possibly pass, so the search sphere is
    bounded by that rather than by the brief's 2 m - a real saving, since the candidate pool is
    the whole elevated cloud. Candidates are tested against their ``search_neighbors`` nearest
    seeds rather than every seed within reach: a farther seed can in principle have the heading
    that puts a point inside its own cylinder, so testing only the single nearest would miss a
    few, while all of them would be a Python loop over pairs.

    Headings are fitted once, from the confirmed set at entry, so the result does not depend on
    the order the points happen to be visited in. Sources come from ``absorb_classes``, which is
    what makes this an override of the grey/pink labels rather than a pole or noise filter; the
    caller has already kept poles out of the candidate pool.

    ``max_points`` bounds the pass, and the budget is spent **farthest-along first**, with the
    perpendicular distance as the tie-break. That ordering is the whole point of a budget here:
    a candidate sitting right beside a confirmed conductor adds nothing, because the cable is
    already there, whereas one at the far end of the along-range is the one closing a hole. On
    the large tile, spending the same budget perpendicular-first instead reached only
    0.40 % of cable-band gaps over 1 m against 0.08 % for the ranking below, and did not move
    the worst gap at all. Without a bound at all the stage inherits whatever the conductor
    class already is and dilates it, exactly as the seed expansion does - 2.1 M points there,
    three quarters of them blobs rather than wire.
    """
    e = spec["inline_cleanup"]
    stats = {"enabled": bool(e["enabled"]), "seeds": 0, "candidates": int(len(candidate_idx)),
             "k_neighbors": int(e["k_neighbors"]), "search_neighbors": int(e["search_neighbors"]),
             "perp_radius": float(e["perp_radius"]), "along_range": float(e["along_range"]),
             "min_hag": float(e["min_hag"]), "chunks": 0, "absorbed": 0,
             "max_points": int(e["max_points"]), "capped": False,
             "heading_elongation_median": 0.0, "absorb_classes": [int(v) for v in e["absorb_classes"]]}
    newly = np.zeros(len(x), dtype=bool)
    if not bool(e["enabled"]) or cKDTree is None or len(candidate_idx) == 0:
        return newly, stats

    seeds = np.flatnonzero(np.asarray(conductor_mask, dtype=bool))
    stats["seeds"] = int(len(seeds))
    if len(seeds) < max(int(e["min_seeds"]), 2):
        return newly, stats

    perp_r = float(e["perp_radius"])
    along_r = float(e["along_range"])
    seek = float(np.hypot(perp_r, along_r))        # nothing farther out can be inside the cylinder
    stats["seek_radius"] = seek
    coords = np.column_stack((x, y, z_raw))
    seed_pts = coords[seeds]

    # one heading per confirmed conductor point: the principal axis of its nearest conductors
    k = max(2, min(int(e["k_neighbors"]), len(seeds)))
    tree = _tree(seed_pts, leafsize=16)
    _, nbr = tree.query(seed_pts, k=k, workers=-1)
    nbr = np.asarray(nbr).reshape(len(seeds), k)
    local = seed_pts[nbr]
    centred = local - local.mean(axis=1, keepdims=True)
    cov = np.einsum("mki,mkj->mij", centred, centred) / max(k - 1, 1)
    evals, evecs = np.linalg.eigh(cov)             # ascending eigenvalues
    headings = evecs[:, :, -1]
    stats["heading_elongation_median"] = float(np.median(
        (evals[:, 2] - evals[:, 1]) / np.maximum(evals[:, 2], 1e-12)))

    # pass one: how far off the axis is each candidate's best shot, and does it also sit
    # within the along-range? Only these two numbers per candidate need to be kept, which is
    # what lets the budget be spent perpendicular-first without holding the pairs.
    kn = max(1, int(e["search_neighbors"]))
    chunk = max(1, int(e["chunk"]))
    best_perp = np.full(len(candidate_idx), np.inf, dtype=np.float32)
    best_along = np.zeros(len(candidate_idx), dtype=np.float32)
    for start in range(0, len(candidate_idx), chunk):
        sub = candidate_idx[start:start + chunk]
        _, hit = tree.query(coords[sub], k=kn, distance_upper_bound=seek, workers=-1)
        hit = np.asarray(hit).reshape(len(sub), kn)
        stats["chunks"] += 1
        ok = hit < len(seeds)                      # the sentinel is "nothing within reach"
        if not ok.any():
            continue
        rows = np.repeat(np.arange(len(sub)), kn).reshape(len(sub), kn)[ok]
        cols = hit[ok]
        vec = coords[sub[rows]] - seed_pts[cols]
        vec_dir = headings[cols]
        along = np.einsum("ij,ij->i", vec, vec_dir)
        perp = vec - along[:, None] * vec_dir
        # fold the k seeds back onto the candidate each pair belongs to
        perp_block = np.full((len(sub), kn), np.inf, dtype=np.float32)
        perp_block[ok] = np.where(np.abs(along) <= along_r,
                                  np.linalg.norm(perp, axis=1), np.inf)
        along_block = np.zeros((len(sub), kn), dtype=np.float32)
        along_block[ok] = along
        pick = np.argmin(perp_block, axis=1)
        here_perp = perp_block[np.arange(len(sub)), pick]
        here_along = along_block[np.arange(len(sub)), pick]
        better = here_perp < best_perp[start:start + chunk]
        best_perp[start:start + chunk] = np.where(better, here_perp,
                                                  best_perp[start:start + chunk])
        best_along[start:start + chunk] = np.where(better, here_along,
                                                   best_along[start:start + chunk])

    inside = np.flatnonzero(best_perp <= perp_r)
    stats["inside_cylinder"] = int(len(inside))
    if not len(inside):
        return newly, stats
    # farthest-along first (that is where a hole is), perpendicular distance as the tie-break
    order = np.lexsort((best_perp[inside], -np.abs(best_along[inside])))
    picked = candidate_idx[inside[order]]
    budget = stats["max_points"]
    if budget > 0 and len(picked) > budget:
        picked = picked[:budget]
        stats["capped"] = True
    newly[picked] = True
    stats["absorbed"] = int(newly.sum())
    return newly, stats


def _isolated_high_noise(classes, z_rel, z_raw, x, y, spec, provenance, pylon_top_z,
                         tree, protected=None):
    """Lock lone high returns to the noise container *before* any wire can claim them.

    A spike - a bird, an atmospheric return, multipath - passes a linearity test all by itself:
    the covariance of a single point's neighbourhood is degenerate, and a short floating
    fragment looks exactly like a piece of cable. Filtering on density first removes that false
    positive at the source, because the wire rules run afterwards and never see the point again.

    Two elevation triggers, per the contract: above ``min_hag``, or above the tallest pylon plus
    ``pylon_z_margin``. A point must also be lonely - fewer than ``max_neighbors`` returns within
    ``radius``, counting the point itself, exactly as a ``query_ball_point`` length does.
    """
    o = spec["isolated_noise"]
    stats = {"enabled": bool(o["enabled"]), "radius": float(o["radius"]),
             "max_neighbors": int(o["max_neighbors"]), "min_hag": float(o["min_hag"]),
             "pylon_z_margin": float(o["pylon_z_margin"]), "candidates": 0, "flagged": 0,
             "flagged_by_hag": 0, "flagged_above_pylons": 0,
             "pylon_top_z": None if pylon_top_z is None else float(pylon_top_z),
             "cap_z": None if pylon_top_z is None else float(pylon_top_z) +
                      float(o["pylon_z_margin"])}
    flagged = np.zeros(len(z_rel), dtype=bool)
    if not bool(o["enabled"]) or tree is None:
        return flagged, stats

    high_hag = z_rel > stats["min_hag"]
    high_z = (np.zeros(len(z_rel), dtype=bool) if stats["cap_z"] is None
              else z_raw > stats["cap_z"])
    protect = np.isin(classes, tuple(int(c) for c in o["preserve_classes"]))
    if protected is not None:
        protect = protect | protected
    protect = protect | np.isin(classes, NOISE_CODES)   # already noise: leave it alone
    candidates = np.flatnonzero((high_hag | high_z) & ~protect)
    stats["candidates"] = int(len(candidates))
    if not len(candidates):
        return flagged, stats

    counts = np.asarray(tree.query_ball_point(
        np.column_stack((x[candidates], y[candidates], z_raw[candidates])),
        r=stats["radius"], return_length=True, workers=-1))
    lone = counts < stats["max_neighbors"]
    picked = candidates[lone]
    stats["flagged"] = int(len(picked))
    stats["flagged_by_hag"] = int(np.count_nonzero(lone & high_hag[candidates]))
    stats["flagged_above_pylons"] = int(np.count_nonzero(lone & high_z[candidates]
                                                       & ~high_hag[candidates]))
    if len(picked):
        provenance[picked] = ORIGIN_HIGH_NOISE
        classes[picked] = 18          # internal high noise -> client 51 (Low point)
        flagged[picked] = True
    return flagged, stats


# ------------------------------------------------------------------------------------ rules
def rule_a_vegetation_split(classes, z_rel, spec, provenance):
    """Split AI vegetation into low/medium/high by HAG and drop sub-threshold vegetation."""
    v = spec["vegetation_split"]
    lo_max, med_max, spur_min = float(v["low_veg_max_hag"]), float(v["med_veg_max_hag"]), \
        float(v["spurious_veg_min_hag"])
    target = int(v["spurious_veg_target_class"])

    is_veg = np.isin(classes, VEG_CODES)
    # Bands are (spur, lo], (lo, med], (med, inf): ground <= 0.15 < low <= 1.0 < medium <= 3.0
    # < high with the current contract. Rules I and J stratify with the same edges.
    low = is_veg & (z_rel > spur_min) & (z_rel <= lo_max)
    med = is_veg & (z_rel > lo_max) & (z_rel <= med_max)
    high = is_veg & (z_rel > med_max)
    spurious = is_veg & (z_rel <= spur_min)

    provenance[low | med | high] = ORIGIN_VEG_SPLIT
    provenance[spurious] = ORIGIN_SPURIOUS_VEG
    stats = {
        "matched": {3: int(low.sum()), 4: int(med.sum()), 5: int(high.sum())},
        "veg_total": int(is_veg.sum()),
        "low": _assign(classes, low, 3),
        "medium": _assign(classes, med, 4),
        "high": _assign(classes, high, 5),
        "spurious": _assign(classes, spurious, target),
        "spurious_target": target,
        "defaulted_mask": spurious.copy(),
    }
    print(f"        [Rule A] Vegetation in: {stats['veg_total']:,} pts | bands "
          f"{spur_min:g}-{lo_max:g}-{med_max:g} m -> "
          f"3: {stats['matched'][3]:,} ({stats['low']:,} updated), "
          f"4: {stats['matched'][4]:,} ({stats['medium']:,} updated), "
          f"5: {stats['matched'][5]:,} ({stats['high']:,} updated)")
    print(f"        [Rule A] Spurious veg <= {spur_min:g} m -> Class {target}: "
          f"{int(spurious.sum()):,} pts ({stats['spurious']:,} updated)")
    return stats


def rule_b_building_recovery(classes, z_rel, x, y, spec, intensity=None,
                             return_number=None, number_of_returns=None, provenance=None):
    """Recover elevated, planar, non-vegetation clusters the colour-blind model suppressed."""
    b = spec["building_recovery"]
    cell = float(b["grid_size"])
    max_std = float(b["max_z_std"])
    min_hag, max_hag = float(b["min_hag"]), float(b["max_hag"])
    min_pts = int(b["min_cell_points"])
    max_veg_frac = float(b["max_veg_fraction"])

    candidates = np.isin(classes, (1, 2)) & (z_rel >= min_hag) & (z_rel <= max_hag)
    print(f"        [Rule B] Elevated candidates (class 1/2, {min_hag:g}-{max_hag:g} m): "
          f"{int(candidates.sum()):,}")
    stats = {"candidates": int(candidates.sum())}

    idx = np.flatnonzero(candidates)
    inverse, cells = _grid_cells(x, y, cell)
    if len(idx):
        # Planarity is measured over the candidate cluster inside each cell: mixing terrain,
        # canopy and roof into one cell statistic would measure heterogeneity, not planarity.
        cnt = np.bincount(inverse[idx], minlength=cells)
        s1 = np.bincount(inverse[idx], weights=z_rel[idx], minlength=cells)
        s2 = np.bincount(inverse[idx], weights=z_rel[idx] ** 2, minlength=cells)
        mean = np.divide(s1, cnt, out=np.zeros(cells), where=cnt > 0)
        var = np.divide(s2, cnt, out=np.zeros(cells), where=cnt > 0) - mean ** 2
        cell_std = np.sqrt(np.maximum(var, 0.0))
        veg_cnt = np.bincount(inverse[idx], weights=np.isin(classes[idx], VEG_CODES).astype(float),
                              minlength=cells)
        veg_frac = np.divide(veg_cnt, cnt, out=np.zeros(cells), where=cnt > 0)
    else:
        cnt = np.zeros(cells)
        cell_std = np.zeros(cells)
        veg_frac = np.zeros(cells)

    # Vegetation check uses the full cell population (canopy may sit outside the band).
    all_cnt = np.bincount(inverse, minlength=cells).astype(np.float64)
    all_veg = np.bincount(inverse, weights=np.isin(classes, VEG_CODES).astype(float),
                          minlength=cells)
    cell_veg_frac = np.divide(all_veg, all_cnt, out=np.zeros(cells), where=all_cnt > 0)

    planar_cell = (cnt >= min_pts) & (cell_std <= max_std)
    non_veg_cell = cell_veg_frac < max_veg_frac
    stage = candidates & planar_cell[inverse] & non_veg_cell[inverse]
    stats["planar_cells"] = int(np.count_nonzero(planar_cell))
    stats["planar_after_veg"] = int(np.unique(inverse[stage]).size) if stage.any() else 0
    stats["dropped_by_planarity"] = int(np.count_nonzero(candidates & ~planar_cell[inverse]))
    stats["dropped_by_vegetation"] = int(np.count_nonzero(candidates & planar_cell[inverse]
                                                          & ~non_veg_cell[inverse]))
    print(f"        [Rule B] Planarity ({cell:g} m cells, candidate HAG std <= {max_std:g} m, "
          f">= {min_pts} pts/cell): {stats['planar_cells']:,} cells qualify, "
          f"-{stats['dropped_by_planarity']:,} pts")
    print(f"        [Rule B] Non-vegetation gate (cell veg fraction < {max_veg_frac:g}): "
          f"-{stats['dropped_by_vegetation']:,} pts ({int(stage.sum()):,} pts on "
          f"{stats['planar_after_veg']:,} clusters left)")

    # --- optional return gate: only usable when the file actually populates returns ---
    if return_number is not None and number_of_returns is not None:
        if int(max(return_number.max(), number_of_returns.max())) <= 0:
            stats["returns_usable"] = False
            print("        [Rule B] [!] ReturnNumber/NumberOfReturns are all zero "
                  "(single-return or unpopulated header)\n"
                  "                  -> first/single-return gate skipped, geometry decides.")
        else:
            stats["returns_usable"] = True
            first = (return_number == 1) | (number_of_returns == 1)
            removed = int(np.count_nonzero(stage & ~first))
            stage &= first
            print(f"        [Rule B] First/single-return gate: -{removed:,} pts "
                  f"({int(stage.sum()):,} left)")

    # --- optional intensity gate (contract: force_intensity) ---
    stats["intensity_applied"] = False
    if intensity is not None:
        ground = classes == 2
        cand_med = float(np.median(intensity[stage])) if stage.any() else 0.0
        ground_med = float(np.median(intensity[ground])) if ground.any() else 0.0
        force = bool(b["force_intensity"])
        if force or cand_med >= ground_med:
            pct = float(b["intensity_percentile"])
            threshold = float(np.percentile(intensity, pct))
            before = int(stage.sum())
            stage &= intensity >= threshold
            stats["intensity_applied"] = True
            print(f"        [Rule B] Intensity gate (>= p{pct:g} = {threshold:,.0f}, "
                  f"{'forced' if force else 'auto'}): -{before - int(stage.sum()):,} pts "
                  f"({int(stage.sum()):,} left)")
        else:
            print(f"        [Rule B] Intensity gate skipped (auto): candidates are dimmer than "
                  f"ground (median {cand_med:,.0f} vs {ground_med:,.0f}), so intensity does not\n"
                  "                  separate structures from terrain in this file. Set "
                  "building_recovery.force_intensity=true to apply it anyway.")

    if provenance is not None:
        provenance[stage] = ORIGIN_BUILDING
    stats["recovered"] = _assign(classes, stage, 6)
    print(f"        [Rule B] Reclassified {stats['recovered']:,} planar, elevated, "
          "non-vegetation points -> Class 6 (Building)")
    return stats


def rule_c_outliers(classes, z_rel, x, y, z_raw, spec, provenance):
    """Contract + spatial HAG outlier detection (no tile-wide absolute Z cut)."""
    o = spec["outlier_detection"]
    preserve = tuple(int(c) for c in o["preserve_classes"])
    neg_hag = float(o["negative_hag_threshold"])
    max_hag = float(o["max_allowable_hag"])
    protected = np.isin(classes, preserve) if preserve else np.zeros(len(classes), dtype=bool)

    low_abs = z_rel < neg_hag
    high_abs = z_rel > max_hag
    stats = {"low_abs": int(low_abs.sum()), "high_abs": int(high_abs.sum()),
             "low_spatial": 0, "high_spatial": 0, "preserve": list(preserve)}

    print(f"        [Rule C] HAG bounds -> low noise < {neg_hag:g} m: {stats['low_abs']:,} pts | "
          f"high noise > {max_hag:g} m: {stats['high_abs']:,} pts")

    low = low_abs.copy()
    high = high_abs.copy()
    if bool(o["use_spatial_hag"]):
        window = float(o["spatial_window_size"])
        pct = float(o["spatial_envelope_percentile"])
        sigma = float(o["spatial_sigma"])
        floor = float(o["spatial_scale_floor"])
        voxel = float(o["isolation_voxel_size"])
        max_neigh = int(o["isolation_max_neighbors"])

        inverse, n_cells = _grid_cells(x, y, window)
        _, scale, envelope_high, envelope_low = _group_hag_stats(z_rel, inverse, n_cells, pct)
        scale = np.maximum(scale, floor)
        envelope_high = envelope_high + sigma * scale
        envelope_low = envelope_low - sigma * scale

        neighbours = _voxel_counts(x, y, z_raw, voxel)
        isolated = neighbours <= max_neigh
        high_spatial = isolated & (z_rel > envelope_high[inverse])
        low_spatial = isolated & (z_rel < envelope_low[inverse])
        low |= low_spatial
        high |= high_spatial
        stats["low_spatial"] = int(low_spatial.sum())
        stats["high_spatial"] = int(high_spatial.sum())
        frac_iso = float(isolated.mean())
        print(f"        [Rule C] Spatial HAG detector ({window:g} m window, p{pct:g} envelope "
              f"+/- {sigma:g} x max(MAD*1.4826, {floor:g} m), isolated voxels "
              f"<= {max_neigh}/{(1/voxel):g} m):\n"
              f"                  {stats['high_spatial']:,} high and {stats['low_spatial']:,} low "
              f"spatial outliers | median local envelope "
              f"{np.nanmedian(envelope_low):.2f} .. {np.nanmedian(envelope_high):.2f} m "
              f"over {n_cells:,} windows")
        if frac_iso > 0.05:
            print(f"        [Rule C] [!] {frac_iso * 100:.1f}% of points are isolated at "
                  f"{voxel:g} m voxels - unusually sparse for this density; the isolation test "
                  "may be too tight.")
    else:
        print("        [Rule C] Spatial HAG detector disabled by contract "
              "(outlier_detection.use_spatial_hag=false)")

    flagged_low = low & ~protected
    flagged_high = high & ~protected
    spared_low = low & protected
    spared_high = high & protected
    stats["preserved_high"] = int(spared_high.sum())
    stats["preserved_low"] = int(spared_low.sum())
    provenance[flagged_low] = ORIGIN_LOW_NOISE
    provenance[flagged_high] = ORIGIN_HIGH_NOISE
    stats["low_noise"] = _assign(classes, flagged_low, 7)
    stats["high_noise"] = _assign(classes, flagged_high, 18)
    print(f"        [Rule C] Marked -> Class 7 (low): {int(flagged_low.sum()):,} | "
          f"Class 18 (high): {int(flagged_high.sum()):,}")

    if preserve:
        spared = spared_low | spared_high
        if spared.any():
            per_class = ", ".join(f"{class_name(c)} {int(np.count_nonzero(spared & (classes == c)))}"
                                  for c in preserve)
            print(f"        [Rule C] {int(spared.sum()):,} pts exempt via preserve_classes "
                  f"{list(preserve)}: {per_class}")
        else:
            print(f"        [Rule C] preserve_classes {list(preserve)} had no matching candidates")

    # Evidence line: what the retired tile-wide Z cut would have done to this file.
    mean_z, std_z = float(np.mean(z_raw)), float(np.std(z_raw))
    legacy = (z_raw < mean_z - 4.5 * std_z) | (z_raw > mean_z + 5.0 * std_z)
    legacy_prot = int(np.count_nonzero(legacy & protected))
    if legacy.any():
        print(f"        [Rule C] Contrast: the retired tile-wide Z cut (+/-4.5/5 sigma on "
              f"absolute Z) would have flagged {int(legacy.sum()):,} pts,\n"
              f"                  including {legacy_prot:,} protected structural points - the "
              "spatial HAG rules keep those in place.")
    return stats


def rule_f_network_extraction(classes, z_rel, x, y, z_raw, spec, provenance):
    """Rule F: recover overhead conductors and poles/towers from local 3D covariance.

    Overhead conductors are long, thin, horizontal spans, so inside the covariance of a
    point's k nearest neighbours the first eigenvalue dominates *and* its eigenvector lies in
    the XY plane: high linearity, low verticality. Poles and pylons are the opposite - a
    vertical string of points - so their principal eigenvector is close to vertical. Both tests
    are restricted to the classes the colour-blind model misfiles this geometry into
    (``network_extraction.source_classes``), so canopy is never claimed as a wire.

    No density or intensity assumption is needed, and the neighbourhood never crosses tile
    boundaries wrongly because it is taken from this cloud only. Note the poles finder also
    runs a structure test (``pole_cell_size`` / ``pole_min_cluster_points`` /
    ``pole_min_vertical_span``): a single point that happens to sit in a vertical string of two
    neighbours is not a pole, so only points that belong to a tall thin cluster are claimed.

    The rule runs in eight ordered stages, because each one needs the previous one's answer:
    poles (which define the pylon top) -> isolated high noise (which must be gone before any
    wire can claim it) -> conductor seeds (guardrails: local density, and a height cap relative
    to that pylon top) -> catenary span propagation (which grows each surviving dash into the
    whole span it belongs to) -> pole-to-pole corridors (which interpolate the space between two
    pole heads, and are themselves carried ``outward_span`` metres past a terminal pole so the
    line is followed off the edge of the survey) -> gap bridge -> pole
    columns (which must not swallow a wire) -> seed expansion (which finally just follows the
    cable through its own 3D neighbourhood, twice). The
    pylon-relative stages are also why this rule stays ahead of Rule C in the driver: Rule C's
    ``preserve_classes`` then protects the wires and poles this rule just found.
    """
    f = spec["network_extraction"]
    if not bool(f["enabled"]):
        print("        [Rule F] Disabled by contract (network_extraction.enabled=false)")
        return {"enabled": False, "conductor_points": 0, "pole_points": 0}
    if cKDTree is None:  # pragma: no cover - depends on the environment
        print(f"        [Rule F] [!] scipy is unavailable ({_SCIPY_ERROR}); conductor and pole\n"
              "                  recovery skipped - those points keep their current class.")
        return {"enabled": False, "conductor_points": 0, "pole_points": 0,
                "error": str(_SCIPY_ERROR)}

    k = int(f["k_neighbors"])
    min_hag = float(f["min_hag"])
    lin_cond = float(f["conductor_linearity"])
    vert_cond = float(f["conductor_max_verticality"])
    min_neighbours = int(f["conductor_min_neighbors"])
    neighbour_radius = float(f["conductor_neighbor_radius"])
    pylon_margin = float(f["conductor_pylon_z_margin"])
    vert_pole = float(f["pole_min_verticality"])
    lin_pole = float(f["pole_min_linearity"])
    pole_hag = float(f["pole_min_hag"])
    cell = float(f["pole_cell_size"])
    min_cluster_pts = int(f["pole_min_cluster_points"])
    min_span = float(f["pole_min_vertical_span"])
    conductor_code = int(f["conductor_class"])
    pole_code = int(f["pole_class"])
    sources = tuple(int(c) for c in f["source_classes"])

    coords = np.column_stack((x, y, z_raw))
    finite = np.isfinite(coords).all(axis=1)
    if not finite.all():
        print(f"        [Rule F] [!] {int((~finite).sum()):,} points have non-finite "
              "coordinates; they are excluded from the analysis and can never match.")
    # Only points the contract can actually claim need a covariance: conductors are seeded from
    # min_hag, poles from pole_min_hag, and claiming is limited to source_classes. Running the
    # neighbour search on that subset rather than on every point of the tile is what keeps this
    # rule fast on a multi-million-point cloud. Wire geometry sitting in the other classes is
    # still reported, from a bounded, evenly strided sample.
    bridge_cfg = spec["conductor_bridge"]
    noise = np.isin(classes, NOISE_CODES)
    in_source_all = np.isin(classes, sources)
    floor = min(min_hag, pole_hag)
    analysis = np.flatnonzero(finite & in_source_all & (z_rel >= floor) & ~noise)
    if len(analysis) < max(k, 2):
        print(f"        [Rule F] Only {len(analysis):,} claimable points above {floor:g} m HAG "
              f"(k_neighbors={k}); conductor/pole recovery skipped.")
        return {"enabled": True, "conductor_points": 0, "pole_points": 0,
                "analysis_points": int(len(analysis))}
    elsewhere = np.flatnonzero(finite & ~in_source_all & (z_rel >= min_hag) & ~noise)
    cap = int(bridge_cfg["diagnostic_cap"])
    diag_step = 1 if len(elsewhere) <= cap else int(np.ceil(len(elsewhere) / max(cap, 1)))
    diag = elsewhere[::diag_step] if diag_step > 1 else elsewhere

    lap = _Laps()
    t0 = time.time()
    linearity = np.zeros(len(classes), dtype=np.float64)
    verticality = np.zeros(len(classes), dtype=np.float64)
    tree = _tree(coords)   # built once, shared by both queries
    lin, vert = _local_geometry(coords, analysis, k, tree=tree)
    linearity[analysis] = lin
    verticality[analysis] = vert
    diag_lin = diag_vert = None
    if len(diag):
        diag_lin, diag_vert = _local_geometry(coords, diag, k, tree=tree)
    print(f"        [Rule F] Local 3D covariance of the {len(analysis):,} claimable pts "
          f"(HAG >= {floor:g} m, k={k}, scipy cKDTree) in {time.time() - t0:.2f} s"
          + (f", plus {len(diag):,} sampled pts from the other classes" if len(diag) else ""))

    before = classes.copy()
    in_source = np.isin(before, sources)
    # Geometric candidates only. The continuity guardrails and the pylon height cap cannot be
    # evaluated yet - a pylon sets the height a wire may plausibly sit at - and the isolated
    # noise pass has to run before the wires, so both are applied further down, in that order:
    #   1. poles (defines the pylon top), 2. isolated noise, 3. conductors, 4. catenary spans,
    #   5. pole-to-pole corridors, 6. gap bridge, 7. pole columns, 8. seed expansion.
    conductor_like = (linearity > lin_cond) & (verticality < vert_cond) & (z_rel >= min_hag)
    pole_like = (verticality >= vert_pole) & (linearity >= lin_pole) & ~conductor_like
    geometric_conductors = conductor_like & in_source
    pole = pole_like & in_source
    stats = {
        "enabled": True, "analysis_points": int(len(analysis)), "k_neighbors": k,
        "source_classes": list(sources),
        "conductor_class": conductor_code, "pole_class": pole_code,
        "conductor_thresholds": [lin_cond, vert_cond, min_hag],
        "conductor_guardrail": [min_neighbours, neighbour_radius, pylon_margin],
        "conductor_like_total": int(conductor_like.sum()),
        "conductor_geometric_candidates": int(geometric_conductors.sum()),
        "pole_like_total": int(pole_like.sum()),
    }

    per_source = {int(c): int(np.count_nonzero(geometric_conductors & (before == c)))
                  for c in sources}
    stats["conductor_by_source"] = per_source
    detail = " | ".join(f"class {c} {class_name(c)}: {per_source[c]:,}" for c in sources)
    print(f"        [Rule F] Conductor geometry (linearity > {lin_cond:g}, verticality < "
          f"{vert_cond:g}, HAG >= {min_hag:g} m) among classes {list(sources)}: "
          f"{int(geometric_conductors.sum()):,} candidates before the guardrails")
    print(f"                  {detail}")
    if diag_lin is not None:
        diag_like = (diag_lin > lin_cond) & (diag_vert < vert_cond) & (z_rel[diag] >= min_hag)
        stats["conductor_diagnostic_sample"] = int(len(diag))
        stats["conductor_diagnostic_points"] = int(diag_like.sum())
        if diag_like.any():
            hit_idx = diag[diag_like]
            codes, counts = np.unique(before[hit_idx], return_counts=True)
            top = sorted(zip(counts.tolist(), codes.tolist()), reverse=True)[:3]
            medians = {int(c): float(np.median(z_rel[hit_idx[before[hit_idx] == int(c)]]))
                       for _, c in top}
            shape = ", ".join(
                f"class {int(c)} {class_name(c)}: {n:,} (median HAG {medians[int(c)]:.1f} m)"
                for n, c in top)
            print(f"                  {int(diag_like.sum()):,} of the {len(diag):,} sampled pts "
                  f"in other classes are wire-like ({shape}); add those codes to "
                  "network_extraction.source_classes to claim them.")
            stats["conductor_like_outside_sources"] = {
                int(c): int(n) for n, c in zip(counts.tolist(), codes.tolist())}

    lap("geometry")
    # --- poles / towers: per-point verticality, then the vertical-structure gate ---
    gated = min_span > 0.0 or min_cluster_pts > 1
    if gated:
        keep, span, _ = _vertical_structure_mask(
            x, y, z_rel, pole, cell, min_cluster_pts, min_span)
        dropped = int(pole.sum() - keep.sum())
        pole = keep
    else:
        span = np.zeros(len(classes))
        dropped = 0
    # The seeds the gate kept, kept separately for reporting: everything below may add points
    # that never were seeds (that is the point of the gap bridge and the column flood), and
    # mixing the two would make the structure statistics meaningless.
    gated_pole = pole.copy()
    # Both guardrails below are measured against the tallest pylon. With no pylon in the tile the
    # cap is dropped rather than guessed, and the log says so.
    pylon_top_z = float(z_raw[gated_pole].max()) if gated_pole.any() else None

    lap("poles")
    # --- stage 2: isolated high noise, before any wire is claimed ---
    noise_flagged, noise_stats = _isolated_high_noise(
        classes, z_rel, z_raw, x, y, spec, provenance, pylon_top_z, tree,
        protected=gated_pole)
    stats["isolated_noise"] = noise_stats
    if noise_stats["enabled"]:
        where = ("no pylon found, so only the HAG trigger is active" if noise_stats["cap_z"] is None
                 else f"above Z {noise_stats['cap_z']:.1f} m (pylon top + "
                      f"{noise_stats['pylon_z_margin']:g} m)")
        print(f"        [Rule F] Isolated noise first (fewer than {noise_stats['max_neighbors']} "
              f"pts within {noise_stats['radius']:g} m, and HAG > {noise_stats['min_hag']:g} m "
              f"or {where}): {noise_stats['flagged']:,} lone pts of "
              f"{noise_stats['candidates']:,} high candidates locked to internal 18 "
              f"({class_name(18)}), delivered as "
              f"{_delivered_label(spec, 18)}")

    lap("noise")
    # --- stage 3: conductor seeds, with the continuity guardrails ---
    cap_z = None if pylon_top_z is None else pylon_top_z + pylon_margin
    keep_z = np.ones(len(classes), dtype=bool) if cap_z is None else (z_raw <= cap_z)
    guarded = np.zeros(len(classes), dtype=bool)
    candidates = np.flatnonzero(conductor_like & in_source & ~noise_flagged)
    dropped_density = dropped_cap = 0
    if len(candidates):
        dense = np.asarray(tree.query_ball_point(
            coords[candidates], r=neighbour_radius, return_length=True, workers=-1)) >= min_neighbours
        under_cap = keep_z[candidates]
        guarded[candidates[dense & under_cap]] = True
        dropped_density = int(np.count_nonzero(~dense))
        dropped_cap = int(np.count_nonzero(~under_cap))
    conductor = guarded
    # The seeds are the conductor set with the least doubt attached - they passed the covariance
    # geometry and the continuity guardrails, and on the labelled tile 97 % of them are really
    # wire. The fallback stages below grow onto far weaker evidence, so the pole validation
    # judges its columns against these rather than against the finished set, whose own spread
    # would make "the wire's height" meaningless exactly when the fallbacks ran away.
    conductor_seeds = conductor.copy()
    stats["conductor_seeds"] = int(conductor.sum())
    stats["conductor_dropped_low_density"] = dropped_density
    stats["conductor_dropped_above_pylon_cap"] = dropped_cap
    stats["pylon_top_z"] = pylon_top_z
    stats["conductor_cap_z"] = cap_z
    print(f"        [Rule F] Conductor guardrails (>= {min_neighbours} pts within "
          f"{neighbour_radius:g} m"
          + (f", Z <= pylon top + {pylon_margin:g} m = {cap_z:.1f} m" if cap_z is not None
             else ", no pylon height cap: no pylon found")
          + f"): {int(conductor.sum()):,} of {len(candidates):,} candidates kept "
          f"({dropped_density:,} not part of a span, {dropped_cap:,} above the cap)")

    lap("seeds")
    # --- stage 4: grow each dash into the whole catenary span it belongs to ---
    # This runs before the gap bridge on purpose: the bridge pairs seeds a few metres apart and
    # immediately fills the tubes between them, which welds the dashes into blobs and robs this
    # stage of the shape it needs to read. Reading the spans first keeps the geometry that the
    # elongation width gates depend on, and the bridge below is then left to mop up whatever the
    # spans could not resolve. Only the classes the contract lists in
    # catenary_spans.absorb_classes are ever offered, and only above the same HAG floor the seeds
    # use, so a span can complete a wire but cannot reach down into the canopy or pull in a pole.
    span_codes = tuple(int(v) for v in spec["catenary_spans"]["absorb_classes"])
    span_candidates = np.flatnonzero(finite & np.isin(before, span_codes) & ~noise
                                     & ~noise_flagged & ~pole & keep_z & (z_rel >= min_hag))
    span, span_stats = _catenary_span_propagation(
        x, y, z_rel, z_raw, conductor, spec, span_candidates)
    if int(span.sum()):
        conductor |= span
    stats["catenary_spans"] = span_stats

    lap("spans")
    # --- stage 5: interpolate the cable between pole heads ---
    # The cable's own geometry has run out by here - the dashes are as complete as a per-point
    # test can make them - so this stage stops looking at the cable and places the *space* it
    # must occupy instead. Only the classes the contract lists in pole_corridors.absorb_classes
    # are ever offered, which is what "all elevated points that are not a pole or noise" means in
    # the client's code space (internal 1/9, 3/4/5, 6 and 17); ground, poles and noise stay out.
    cor = spec["pole_corridors"]
    corridor_codes = tuple(int(v) for v in cor["absorb_classes"])
    corridor_candidates = np.flatnonzero(finite & np.isin(before, corridor_codes) & ~noise
                                         & ~noise_flagged & ~pole
                                         & (z_rel >= float(cor["min_hag"])))
    corridor, corridor_stats = _pole_to_pole_corridors(
        x, y, z_rel, z_raw, pole, conductor, spec, corridor_candidates)
    if int(corridor.sum()):
        conductor |= corridor
    stats["pole_corridors"] = corridor_stats

    lap("corridors")
    # --- stage 6: bridge the gaps between wire seeds ---
    bridge_candidates = np.flatnonzero(finite & ~noise & ~noise_flagged & (z_rel >= min_hag)
                                       & ~pole & keep_z)
    bridge, bridge_stats = _bridge_conductor_gaps(
        x, y, z_rel, z_raw, conductor, spec, bridge_candidates)
    if int(bridge.sum()):
        conductor |= bridge
    stats["conductor_bridge"] = bridge_stats

    lap("bridge")
    # --- stage 7: flood each pole's column ---
    column, column_stats = _pole_column_fill(x, y, z_rel, pole, spec,
                                             exclude=conductor)
    if int(column.sum()):
        pole |= column
    stats["pole_cylinder"] = column_stats

    lap("columns")
    # --- stage 8: follow the cable through its own neighbourhood, twice ---
    # Last, after the columns have taken what they are going to take, so nothing this pass claims
    # can be swallowed by a pole afterwards. ``~column`` matters because ``before`` still shows
    # the pre-rule classes for the points the column flood just made into poles.
    exp = spec["seed_expansion"]
    expansion_codes = tuple(int(v) for v in exp["absorb_classes"])
    # ``~conductor`` keeps the pool to points that are still unclaimed: ``before`` is the
    # pre-rule snapshot, so without it every point an earlier stage already made into a
    # conductor would be offered here again and counted as newly absorbed.
    expansion_candidates = np.flatnonzero(finite & np.isin(before, expansion_codes) & ~noise
                                          & ~noise_flagged & ~pole & ~column & ~conductor
                                          & (z_rel >= float(exp["min_hag"])))
    expansion, expansion_stats = _expand_conductor_seeds(
        x, y, z_raw, conductor, spec, expansion_candidates)
    if int(expansion.sum()):
        conductor |= expansion
    stats["seed_expansion"] = expansion_stats

    lap("expansion")
    # --- stage 9: override whatever sits inline with the cable, along its own heading ---
    # Last of all, so it sees the most complete conductor set to fit headings from, and after
    # the columns, so a point the flood already made into a pole is not silently re-taken.
    inline = spec["inline_cleanup"]
    inline_codes = tuple(int(v) for v in inline["absorb_classes"])
    inline_candidates = np.flatnonzero(finite & np.isin(before, inline_codes) & ~noise
                                       & ~noise_flagged & ~pole & ~column & ~conductor
                                       & (z_rel >= float(inline["min_hag"])))
    inline_new, inline_stats = _inline_conductor_cleanup(
        x, y, z_raw, conductor, spec, inline_candidates)
    if int(inline_new.sum()):
        conductor |= inline_new
    stats["inline_cleanup"] = inline_stats

    lap("inline")
    # --- stage 10: keep only the columns that look like a structure ---
    # Last, so it judges the finished object: the head it tests is the one the column flood
    # built, and the wire band it tests it against is the final conductor set. Points this
    # rejects are handed back to the class the AI gave them, which is what makes the revert
    # lossless - a canopy point the flood swallowed returns to the AI's own canopy label.
    protected = noise | noise_flagged
    pole, validation_stats = _pole_validate(x, y, z_rel, pole, conductor_seeds, spec,
                                            protected=protected)
    stats["pole_validation"] = validation_stats
    if int(validation_stats["reverted_points"]):
        reverted = column & ~pole
        stats["pole_reverted"] = int(np.count_nonzero(reverted))
        stats["pole_reverted_by_source"] = {
            int(c): int(np.count_nonzero(reverted & (before == c)))
            for c in np.unique(before[reverted])}

    if validation_stats["enabled"]:
        if validation_stats["skipped_no_wire"]:
            print(f"        [Rule F] Pole validation skipped: no conductor was confirmed, so there is "
                  f"no wire height to judge {validation_stats['skipped_no_wire']:,} pole pts against")
        elif validation_stats["wire_band"] is not None:
            lo, hi = validation_stats["wire_band"]
            print(f"        [Rule F] Pole validation (head no more than "
                  f"{validation_stats['head_margin']:g} m above the wire passing within "
                  f"{validation_stats['local_radius']:g} m; wire spans HAG {lo:.1f}-{hi:.1f} m): "
                  f"{validation_stats['kept']:,} of {validation_stats['columns']:,} columns kept"
                  + (f", {validation_stats['skipped_protected']:,} skipped (head is all "
                     "noise or structure)" if validation_stats["skipped_protected"] else "")
                  + f"; {stats.get('pole_reverted', 0):,} pts returned to their AI class")
            if stats.get("pole_reverted_by_source"):
                print("                  returned to: " + " | ".join(
                    f"class {c} {class_name(c)}: {n:,}"
                    for c, n in sorted(stats["pole_reverted_by_source"].items(),
                                       key=lambda kv: -kv[1])[:5]))

    lap("validate")
    stats["pole_points_gated_out"] = dropped
    stats["conductor_points"] = int(conductor.sum())
    stats["pole_points"] = int(pole.sum())
    stats["pole_seed_points"] = int(gated_pole.sum())
    stats["pole_seed_by_source"] = {int(c): int(np.count_nonzero(gated_pole & (before == c)))
                                     for c in sources}
    stats["conductor_client_class"] = int(spec["ptc_class_mapping"].get(conductor_code,
                                                                      conductor_code))
    stats["pole_client_class"] = int(spec["ptc_class_mapping"].get(pole_code, pole_code))
    print(f"        [Rule F] Pole seeds (verticality >= {vert_pole:g}, linearity >= "
          f"{lin_pole:g}, HAG >= {pole_hag:g} m) among classes {list(sources)}: "
          f"{int(gated_pole.sum()):,} pts")
    print("                  " + " | ".join(
        f"class {c} {class_name(c)}: {stats['pole_seed_by_source'][c]:,}" for c in sources))
    if gated:
        cells = int(np.unique(np.floor(np.column_stack((x[gated_pole], y[gated_pole])) / cell),
                              axis=0).shape[0]) if gated_pole.any() else 0
        print(f"                  structure gate ({cell:g} m cells, >= {min_cluster_pts} pts "
              f"spanning >= {min_span:g} m): {cells:,} vertical structures; "
              f"{dropped:,} lone/short vertical specks left in place "
              "(set pole_min_vertical_span=0 to keep them)")
        if gated_pole.any():
            stats["pole_structures"] = cells
            stats["pole_span_median"] = float(np.median(span[gated_pole]))
            print(f"                  structure height: median {np.median(span[gated_pole]):.1f} m, "
                  f"max {span[gated_pole].max():.1f} m; bottoms reach HAG "
                  f"{np.min(z_rel[gated_pole]):.2f} m")

    if span_stats["enabled"]:
        print(f"        [Rule F] Catenary spans: {span_stats['seeds']:,} seeds -> "
              f"{span_stats['nodes']:,} grid nodes, {span_stats['links']:,} links"
              + (" (budget hit: shortest links kept)" if span_stats["links_capped"] else "")
              + f"; {span_stats['spans_kept']:,} of {span_stats['spans']:,} components look "
              f"like a wire ({span_stats['rejected_short']:,} too short, "
              f"{span_stats['rejected_not_elongated']:,} not straight, "
              f"{span_stats['rejected_too_wide']:,} too wide) -> "
              f"{span_stats['samples']:,} path samples; {int(span.sum()):,} pts fall inside a "
              f"{span_stats['radius']:g} m cylinder along them")
        if span_stats["span_length_median"] is not None:
            print(f"                  wire-shaped spans: length median "
                  f"{span_stats['span_length_median']:.1f} m, width median "
                  f"{span_stats['span_width_median']:.2f} m")
        if span.any():
            codes_s, counts_s = np.unique(before[span], return_counts=True)
            order_s = np.argsort(-counts_s, kind="stable")
            stats["span_by_source"] = {int(codes_s[i]): int(counts_s[i]) for i in order_s}
            print("                  absorbed from: " + " | ".join(
                f"class {int(codes_s[i])} {class_name(codes_s[i])}: {int(counts_s[i]):,}"
                for i in order_s[:5]))
    if corridor_stats["enabled"]:
        print(f"        [Rule F] Pole-to-pole corridors: {corridor_stats['tower_points']:,} pole "
              f"pts -> {corridor_stats['towers']:,} pole heads -> "
              f"{corridor_stats['tower_pairs']:,} adjacent pairs within "
              f"{corridor_stats['max_span']:g} m; {corridor_stats['supported']:,} carry wire "
              f"evidence along their length ({corridor_stats['rejected_unsupported']:,} "
              f"unsupported, {corridor_stats['rejected_unstable']:,} with no cable-like height)"
              + (" (budget hit: best-supported corridors kept)"
                 if corridor_stats["corridors_capped"] else ""))
        if corridor_stats["coverage_median"] is not None:
            print(f"                  support: seed coverage median "
                  f"{corridor_stats['coverage_median'] * 100:.0f} %, seed-vs-chord median "
                  f"{corridor_stats['residual_median']:.2f} m; "
                  f"{corridor_stats['samples']:,} samples inside a "
                  f"{corridor_stats['radius']:g} m XY cylinder"
                  + (", sample budget reached" if corridor_stats["samples_capped"] else ""))
            if corridor_stats["outward_span"] > 0.0:
                print(f"                  boundary extrapolation: {corridor_stats['terminal_poles']:,} "
                      f"terminal poles (one neighbour), "
                      f"{corridor_stats['extended_corridors']:,} corridors carried "
                      f"{corridor_stats['outward_span']:g} m outward past the last head")
        if corridor.any():
            codes_c, counts_c = np.unique(before[corridor], return_counts=True)
            order_cor = np.argsort(-counts_c, kind="stable")
            stats["corridor_by_source"] = {int(codes_c[i]): int(counts_c[i])
                                           for i in order_cor}
            print(f"                  {int(corridor.sum()):,} pts forced to internal "
                  f"{conductor_code} ({class_name(conductor_code)}), from: " + " | ".join(
                f"class {int(codes_c[i])} {class_name(codes_c[i])}: {int(counts_c[i]):,}"
                for i in order_cor[:6]))
    if bridge_stats["enabled"]:
        print(f"        [Rule F] Gap bridge: {bridge_stats['pairs']:,} seed pairs within "
              f"{bridge_stats['max_gap']:g} m, {bridge_stats['gaps']:,} of them wider than "
              f"{bridge_stats['min_gap']:g} m -> {bridge_stats['samples']:,} samples along the "
              f"spans; {int(bridge.sum()):,} pts fall inside a span tube"
              + (f" ({'shortest' if bridge_stats['prefer_shortest'] else 'widest'} gaps only: "
                 "pair budget reached)" if bridge_stats["pairs_capped"] else ""))
        if bridge.any():
            codes, counts = np.unique(before[bridge], return_counts=True)
            order_b = np.argsort(-counts, kind="stable")
            stats["bridge_by_source"] = {int(codes[i]): int(counts[i]) for i in order_b}
            print("                  bridge fill came from: " + " | ".join(
                f"class {int(codes[i])} {class_name(codes[i])}: {int(counts[i]):,}"
                for i in order_b[:5]))
    if column_stats["enabled"]:
        column_new = column & ~gated_pole
        print(f"        [Rule F] Pole columns: {column_stats['columns']:,} cylinders "
              f"(R={column_stats['radius']:g} m, HAG {column_stats['base_hag']:g} m to top + "
              f"{column_stats['top_margin']:g} m, >= "
              f"{int(spec['pole_cylinder']['min_seed_points'])} seeds); "
              f"{int(column.sum()):,} pts are inside them, {int(column_new.sum()):,} of which "
              f"were not seeds and are forced to internal {pole_code} "
              f"({class_name(pole_code)})")
        if column_new.any():
            codes, counts = np.unique(before[column_new], return_counts=True)
            order_c = np.argsort(-counts, kind="stable")
            stats["cylinder_by_source"] = {int(codes[i]): int(counts[i]) for i in order_c}
            print("                  removed from: " + " | ".join(
                f"class {int(codes[i])} {class_name(codes[i])}: {int(counts[i]):,}"
                for i in order_c[:5]))
        if column_stats["empty_columns"]:
            print(f"                  {column_stats['empty_columns']:,} columns had no points "
                  "to claim")
    if expansion_stats["enabled"]:
        print(f"        [Rule F] Seed expansion: {expansion_stats['iterations']} rounds of "
              f"{expansion_stats['radius']:g} m 3D growth from the conductor class -> "
              f"{int(expansion.sum()):,} pts absorbed "
              f"(per round {', '.join(f'{v:,}' for v in expansion_stats['per_iteration'])})"
              + (f" (budget {expansion_stats['max_points']:,} hit: nearest candidates kept)"
                 if expansion_stats["capped"] else ""))
        if expansion.any():
            codes_e, counts_e = np.unique(before[expansion], return_counts=True)
            order_e = np.argsort(-counts_e, kind="stable")
            stats["expansion_by_source"] = {int(codes_e[i]): int(counts_e[i]) for i in order_e}
            print("                  absorbed from: " + " | ".join(
                f"class {int(codes_e[i])} {class_name(codes_e[i])}: {int(counts_e[i]):,}"
                for i in order_e[:5]))
    if inline_stats["enabled"]:
        print(f"        [Rule F] Inline cleanup along the cable's own heading: "
              f"{inline_stats['seeds']:,} seeds -> {inline_stats['heading_elongation_median']:.2f} "
              f"median heading elongation; {inline_stats['candidates']:,} candidates "
              f"(HAG >= {inline_stats['min_hag']:g} m) within a "
              f"{inline_stats['perp_radius']:g} m cylinder over "
              f"-{inline_stats['along_range']:g}..+{inline_stats['along_range']:g} m "
              f"(search {inline_stats['seek_radius']:.2f} m), "
              f"{inline_stats.get('inside_cylinder', 0):,} inside the cylinder -> "
              f"{int(inline_new.sum()):,} pts absorbed"
              + (f" (budget {inline_stats['max_points']:,} hit: farthest-along kept)"
                 if inline_stats["capped"] else "")
              + ("" if inline_stats["seeds"] >= int(spec["inline_cleanup"]["min_seeds"])
                 else " (not enough seeds to fit headings)"))
        if inline_new.any():
            codes_i, counts_i = np.unique(before[inline_new], return_counts=True)
            order_i = np.argsort(-counts_i, kind="stable")
            stats["inline_by_source"] = {int(codes_i[i]): int(counts_i[i]) for i in order_i}
            print("                  overrode: " + " | ".join(
                f"class {int(codes_i[i])} {class_name(codes_i[i])}: {int(counts_i[i]):,}"
                for i in order_i[:5]))
    stats["bridge_points"] = int(bridge.sum())
    stats["span_points"] = int(span.sum())
    stats["corridor_points"] = int(corridor.sum())
    stats["expansion_points"] = int(expansion.sum())
    stats["cylinder_points"] = int(column.sum())

    # --- evidence that the two structures belong together ---
    if conductor.any() and pole.any():
        wire_tree = _tree(np.column_stack((x[conductor], y[conductor])))
        dist, _ = wire_tree.query(np.column_stack((x[pole], y[pole])), k=1, workers=-1)
        stats["pole_to_conductor_xy_median"] = float(np.median(dist))
        stats["pole_within_5m_percent"] = float((dist <= 5.0).mean() * 100.0)
        print(f"                  {int((dist <= 5.0).sum()):,} of {len(dist):,} pole pts sit "
              f"within 5 m of a conductor point (median {np.median(dist):.2f} m)")

    provenance[conductor] = ORIGIN_CONDUCTOR
    provenance[pole] = ORIGIN_POLE
    stats["conductor_changed"] = _assign(classes, conductor, conductor_code)
    stats["pole_changed"] = _assign(classes, pole, pole_code)
    print(f"        [Rule F] Reclassified {int(conductor.sum()):,} pts -> internal {conductor_code} "
          f"({class_name(conductor_code)}), delivered as "
          f"{_delivered_label(spec, conductor_code)}")
    print(f"                  {int(pole.sum()):,} pts -> internal {pole_code} "
          f"({class_name(pole_code)}), delivered as {_delivered_label(spec, pole_code)}")
    print(f"        [Rule F] time by stage: {lap.line()}")
    return stats


def _local_shape(cloud: np.ndarray, tree, query_pos: np.ndarray, k: int,
                 chunk: int = 200_000) -> tuple:
    """Planarity and sphericity of every query point's k nearest neighbours in ``cloud``.

    The companion of ``_local_geometry``: that one asks "is this a line", this one asks "is this
    a sheet or a volume". With lambda1 >= lambda2 >= lambda3, planarity = (l2 - l3) / l1 and
    sphericity = l3 / l1; a degenerate neighbourhood scores 0 on both, so it can never pass as a
    crown. Chunked and in fixed order, so it is reproducible and bounded in memory.
    """
    n = len(query_pos)
    planarity = np.zeros(n, dtype=np.float64)
    sphericity = np.zeros(n, dtype=np.float64)
    k_eff = int(min(k, len(cloud)))
    if n == 0 or k_eff < 3:
        return planarity, sphericity
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        _, neighbours = tree.query(cloud[query_pos[start:stop]], k=k_eff, workers=-1)
        evals = _neighbour_eig(cloud, np.asarray(neighbours).reshape(stop - start, k_eff), k_eff)
        l1, l2, l3 = evals[:, 2], evals[:, 1], evals[:, 0]
        ok = np.isfinite(l1) & (l1 > 1e-12)
        block_plan = np.zeros(stop - start, dtype=np.float64)
        block_sph = np.zeros(stop - start, dtype=np.float64)
        block_plan[ok] = (l2[ok] - l3[ok]) / l1[ok]
        block_sph[ok] = np.maximum(l3[ok], 0.0) / l1[ok]
        planarity[start:stop] = block_plan
        sphericity[start:stop] = block_sph
    return planarity, sphericity


def _valid_conductor_lines(coords: np.ndarray, cond_idx: np.ndarray, c: dict) -> tuple:
    """Which conductor points belong to a real, continuous cable - judged piece by piece.

    "Connected Class 14" is not enough: by the time Rule F has finished, a crown it misread can
    be a forty-metre component of a few thousand points. Nor can a real line be judged as one
    object: a distribution line bends at every pole, so on the 123 tile its whole 171 m network is
    a single component that measures 32 m "wide" to a global PCA, and was rejected outright.

    So each component is cut into ``line_piece_size`` cubes and every piece is judged on its own:
    a cable piece is straight (``min_line_elongation``), thin (``max_line_width`` - wide enough
    for a three-phase bundle) and not steep (``max_line_verticality``, which is what refuses the
    tall red column a conifer leaves behind). A component is a line when its straight pieces add
    up to ``min_line_length``; its straight pieces are the valid conductor. The corners and pole
    heads that fail the piece test are then covered by the corridor around their neighbours.
    Measured: on the 123 tile this keeps 98.5 % of the wire above the canopy inside the corridor
    (7.5 % of the tree-row speckle below it), where the whole-component test kept 0 %; on
    FUNIVIA000007 the corridor still covers only ~5 % of the crown speckle.

    Everything runs on a ``line_grid_cell`` decimation of the wire, which bounds the link graph on
    tiles where Rule F returns several hundred thousand conductor points.
    Returns ``(valid_idx, stats)``: indices of the conductor points that passed.
    """
    stats = {"conductor_points": int(len(cond_idx)), "nodes": 0, "components": 0, "pieces": 0,
             "line_pieces": 0, "lines": 0, "valid_points": 0, "line_lengths": []}
    if len(cond_idx) < 2:
        return np.zeros(0, dtype=np.int64), stats
    pts = coords[cond_idx]
    keys = np.floor(pts / float(c["line_grid_cell"])).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    nodes = pts[first]
    stats["nodes"] = int(len(nodes))
    links = cKDTree(nodes).query_pairs(r=float(c["line_link_radius"]), output_type="ndarray")
    labels = _link_labels(len(nodes), links)
    stats["components"] = int(len(np.unique(labels)))

    cube = np.floor(nodes / float(c["line_piece_size"])).astype(np.int64)
    piece = np.asarray(np.unique(np.column_stack((labels, cube)), axis=0,
                                 return_inverse=True)[1]).ravel()
    counts = np.bincount(piece)
    order = np.argsort(piece, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    stats["pieces"] = int(len(counts))
    straight = np.zeros(len(counts), dtype=bool)
    extent = np.zeros(len(counts), dtype=np.float64)
    min_el, max_w = float(c["min_line_elongation"]), float(c["max_line_width"])
    max_vert = float(c["max_line_verticality"])
    for p in np.flatnonzero(counts >= int(c["min_piece_points"])):
        part = nodes[order[starts[p]:starts[p] + counts[p]]]
        centred = part - part.mean(axis=0)
        evals, evecs = np.linalg.eigh(centred.T @ centred / len(part))
        l1 = float(evals[2])
        if l1 <= 1e-12 or (l1 - float(evals[1])) / l1 < min_el:
            continue
        if 2.0 * float(np.sqrt(max(evals[1] + evals[0], 0.0))) > max_w:
            continue
        if abs(float(evecs[2, 2])) > max_vert:
            continue
        along = centred @ evecs[:, 2]
        straight[p] = True
        extent[p] = float(along.max() - along.min())
    piece_label = np.zeros(len(counts), dtype=np.int64)
    piece_label[piece] = labels
    line_length = np.bincount(piece_label[straight], weights=extent[straight],
                              minlength=int(labels.max()) + 1)
    is_line = line_length >= float(c["min_line_length"])
    valid_node = straight[piece] & is_line[labels]
    valid = cond_idx[valid_node[inverse]]
    stats["line_pieces"] = int(np.count_nonzero(straight & is_line[piece_label]))
    stats["lines"] = int(is_line.sum())
    stats["line_lengths"] = sorted((round(float(v), 1) for v in line_length[is_line]),
                                   reverse=True)
    stats["valid_points"] = int(len(valid))
    return valid, stats


def _reject_foliage_lines(coords, classes, z_rel, valid_idx, c) -> tuple:
    """Drop the confirmed lines that run through a tree crown rather than through the air.

    A crown has straight runs in it - a branch, a row of returns along a stem - and once Rule F has
    painted them red, a few pass the piecewise line test. Their corridors then freeze the whole
    crown around them: on DU3039046 93 % of the grey and red left in the canopy was frozen by such
    lines, sitting 0-0.6 m from them. What the line test cannot see is what surrounds a line. A
    real cable runs through air (the 123 line: 16 % of the non-wire points within 1 m are foliage;
    the cable-car span 25 %) or past structures (the cable-car station: 0 %, its surroundings are
    planar), while the false ones sit in a crown (76-98 %).

    So each confirmed line - a connected set of confirmed points, linked as in the detector - is
    judged on the non-wire points within ``line_foliage_radius`` of it: when there are at least
    ``line_foliage_min_points`` of them and at least ``line_foliage_max_share`` are volumetric
    (the crown test's sphericity / planarity thresholds), the line is not a line. Lines with
    fewer neighbours are kept: an empty neighbourhood is exactly what a cable has.
    Returns ``(valid_idx, stats)``.
    """
    stats = {"lines_checked": 0, "lines_rejected": 0, "points_rejected": 0}
    if not len(valid_idx) or z_rel is None:
        return valid_idx, stats
    pts = coords[valid_idx]
    keys = np.floor(pts / float(c["line_grid_cell"])).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    nodes = pts[first]
    links = cKDTree(nodes).query_pairs(r=float(c["line_link_radius"]), output_type="ndarray")
    labels = _link_labels(len(nodes), links)
    conductor_code = int(c.get("_conductor_code", 14))
    cloud_idx = np.flatnonzero(np.isfinite(coords).all(axis=1) & (z_rel > 0.15))
    cloud = coords[cloud_idx]
    tree = _tree(cloud)
    non_wire = classes[cloud_idx] != conductor_code
    keep_label = np.ones(int(labels.max()) + 1, dtype=bool)
    r_f = float(c["line_foliage_radius"])
    l_order, l_start, l_count = _group_members(labels)
    for lab in np.flatnonzero(l_count):
        sel = nodes[l_order[l_start[lab]:l_start[lab] + l_count[lab]]]
        sample = sel[:: max(1, len(sel) // 200)]
        near = np.unique(np.concatenate(
            [np.asarray(q, dtype=np.int64) for q in tree.query_ball_point(sample, r=r_f)]))
        near = near[non_wire[near]]
        stats["lines_checked"] += 1
        if len(near) < int(c["line_foliage_min_points"]):
            continue
        plan, sph = _local_shape(cloud, tree, near, int(c["crown_k_neighbors"]))
        share = float(np.mean((sph >= float(c["min_sphericity"]))
                              & (plan <= float(c["max_planarity"]))))
        if share >= float(c["line_foliage_max_share"]):
            keep_label[lab] = False
            stats["lines_rejected"] += 1
    keep = keep_label[labels[inverse]]
    stats["points_rejected"] = int(np.count_nonzero(~keep))
    return valid_idx[keep], stats


def build_immutable_utility_mask(classes, x, y, z_raw, spec, return_valid=False,
                                 z_rel=None, structure=None, traced=None) -> tuple:
    """The points Rule I may never touch: every pole, every confirmed wire, and its corridor.

    Frozen before Rule I runs, from Rule F's finished (and validated) poles and conductors:
      * every internal pole point (``network_extraction.pole_class``);
      * every conductor point that belongs to a real line (``_valid_conductor_lines``);
      * every point, of any class and at any height, within ``corridor_radius`` of a confirmed
        wire measured horizontally - a pure 2D cylinder, so a wire on a steep slope or in a deep
        catenary sag is covered over its full height.
    Returns ``(mask, stats)``.
    """
    c = spec["canopy_homogenization"]
    conductor_code = int(spec["network_extraction"]["conductor_class"])
    pole_code = int(spec["network_extraction"]["pole_class"])
    coords = np.column_stack((x, y, z_raw))
    finite = np.isfinite(coords).all(axis=1)
    mask = classes == pole_code
    stats = {"pole_points": int(mask.sum())}
    # Buildings are frozen too: every Building point left after Rules L and R is a validated
    # roof, and ``structure`` is the column under the consolidated roofs (walls), which the crown
    # tests otherwise read as foliage - on DU3039046 they turned the upper station green.
    building = classes == 6
    if structure is not None:
        building |= structure
    mask |= building
    stats["structure_points"] = int(building.sum())
    # Rule T's traced poles and wire: certain, wherever the canopy stands around them
    traced_wire = np.zeros(0, dtype=np.int64)
    if traced is not None:
        traced_wire = np.asarray(traced["wire_idx"], dtype=np.int64)
        mask[np.asarray(traced["pole_idx"], dtype=np.int64)] = True
        mask[traced_wire] = True
    if not bool(c["enabled"]) or cKDTree is None:
        return (mask, stats, traced_wire) if return_valid else (mask, stats)

    valid_idx, line_stats = _valid_conductor_lines(
        coords, np.flatnonzero(finite & (classes == conductor_code)), c)
    lengths = line_stats.pop("line_lengths")
    stats["lines"] = line_stats
    c["_conductor_code"] = conductor_code
    valid_idx, foliage_stats = _reject_foliage_lines(coords, classes, z_rel, valid_idx, c)
    c.pop("_conductor_code", None)
    stats["foliage_gate"] = foliage_stats
    print(f"        [Rule I] Line foliage gate (>= {int(c['line_foliage_min_points'])} non-wire pts "
          f"within {float(c['line_foliage_radius']):g} m, >= "
          f"{float(c['line_foliage_max_share']):.0%} of them crown-shaped): "
          f"{foliage_stats['lines_rejected']:,} of {foliage_stats['lines_checked']:,} confirmed lines "
          f"run through a crown and are dropped ({foliage_stats['points_rejected']:,} pts)")
    mask[valid_idx] = True
    corridor = np.zeros(len(classes), dtype=bool)
    if len(valid_idx):
        keys = np.floor(coords[valid_idx] / float(c["line_grid_cell"])).astype(np.int64)
        nodes = coords[valid_idx][np.sort(np.unique(keys, axis=0, return_index=True)[1])]
        query = np.flatnonzero(finite)
        k_nodes = int(min(8, len(nodes)))
        _, hit = cKDTree(nodes[:, :2]).query(coords[query, :2], k=k_nodes,
                                             distance_upper_bound=float(c["corridor_radius"]),
                                             workers=-1)
        hit = np.asarray(hit).reshape(len(query), k_nodes)
        ok = hit < len(nodes)                                  # sentinel: nothing in reach
        depth = c.get("corridor_depth")
        if depth is not None:
            # Measured from the wire directly overhead, not from a tile-wide band, so a span on a
            # steep slope or in a deep sag is covered over its own height. What it stops is the
            # forest 10-100 m under a cable being frozen: on DU3039046 that was 62 k grey and
            # 31 k red points more than 5 m below the wire above them.
            node_z = nodes[np.minimum(hit, len(nodes) - 1), 2]
            ok &= coords[query, 2][:, None] >= node_z - float(depth)
        height = c.get("corridor_height")
        if height is not None:
            # And bounded above: a wire has nothing to protect metres over it (pole heads are
            # frozen as poles anyway). Without this a short false "line" low in a crown froze the
            # whole crown above it - one DU3039046 tree came back red from 6 m to 16 m HAG.
            node_z = nodes[np.minimum(hit, len(nodes) - 1), 2]
            ok &= coords[query, 2][:, None] <= node_z + float(height)
        corridor[query[ok.any(axis=1)]] = True
    mask |= corridor
    stats["corridor_points"] = int(corridor.sum())
    stats["corridor_wire_points"] = int(np.count_nonzero(corridor & (classes == conductor_code)))
    stats["wire_points"] = int(np.count_nonzero(classes == conductor_code))
    stats["immutable_points"] = int(mask.sum())
    print(f"        [Rule I] Utility protection: {line_stats['conductor_points']:,} internal "
          f"{conductor_code} pts -> {line_stats['components']:,} components cut into "
          f"{line_stats['pieces']:,} pieces of {float(c['line_piece_size']):g} m; "
          f"{line_stats['line_pieces']:,} straight pieces form {line_stats['lines']:,} lines "
          f">= {float(c['min_line_length']):g} m holding {line_stats['valid_points']:,} pts"
          + (f" (lengths {lengths[:8]} m)" if lengths else ""))
    depth, height = c.get("corridor_depth"), c.get("corridor_height")
    band = ("all heights" if depth is None and height is None else
            f"{'-' if depth is None else f'{float(depth):g} m under'} .. "
            f"{'-' if height is None else f'{float(height):g} m over'} the wire")
    print(f"                  frozen: {stats['pole_points']:,} pole pts + confirmed wire + "
          f"{stats['corridor_points']:,} pts inside the {float(c['corridor_radius']):g} m 2D "
          f"corridor ({band}; "
          f"{stats['corridor_wire_points']:,} of "
          f"{stats['wire_points']:,} wire pts) = {stats['immutable_points']:,} immutable pts")
    # the traced wire counts as confirmed (Rules J and K read it) but gets no corridor of its own:
    # it is a thin line, and a 1.5 m tube would freeze the junk Rule F left around it
    valid_idx = np.union1d(valid_idx, traced_wire)
    return (mask, stats, valid_idx) if return_valid else (mask, stats)


def purge_low_wire(classes, z_rel, x, y, z_raw, spec, provenance):
    """Wire below ``low_wire_max_hag`` that does not touch a pole is undergrowth, not a cable.

    No conductor hangs 2 m over the ground except where it drops to a pole, so a wire point that
    low with no pole point within ``pole_connect_radius`` (3D) is a shrub Rule F's linearity test
    fired on - the network stages grow wire downwards through the canopy. It goes to vegetation
    by Rule A's bands. Run before the utility mask, so a confirmed line can never protect it.
    """
    c = spec["canopy_homogenization"]
    conductor_code = int(spec["network_extraction"]["conductor_class"])
    pole_code = int(spec["network_extraction"]["pole_class"])
    v = spec["vegetation_split"]
    spur, lo_max = float(v["spurious_veg_min_hag"]), float(v["low_veg_max_hag"])
    low = np.flatnonzero((classes == conductor_code) & (z_rel < float(c["low_wire_max_hag"])))
    stats = {"low_wire": int(len(low)), "kept_at_pole": 0, "purged": 0}
    if not len(low):
        print(f"        [Rule I] Wire below {float(c['low_wire_max_hag']):g} m HAG: none")
        return stats
    coords = np.column_stack((x, y, z_raw))
    pole = np.flatnonzero(classes == pole_code)
    at_pole = np.zeros(len(low), dtype=bool)
    if len(pole) and cKDTree is not None:
        at_pole = np.asarray(cKDTree(coords[pole]).query(
            coords[low], k=1, distance_upper_bound=float(c["pole_connect_radius"]),
            workers=-1)[1]) < len(pole)
    out = low[~at_pole]
    hag = z_rel[out]
    classes[out] = np.where(hag <= spur, 2, np.where(hag <= lo_max, 3, 4)).astype(classes.dtype)
    provenance[out] = ORIGIN_CANOPY
    stats["kept_at_pole"] = int(at_pole.sum())
    stats["purged"] = int(len(out))
    print(f"        [Rule I] Wire below {float(c['low_wire_max_hag']):g} m HAG: {len(low):,} pts, "
          f"{stats['kept_at_pole']:,} touch a pole (within {float(c['pole_connect_radius']):g} m) "
          f"and stay; {stats['purged']:,} -> vegetation")
    return stats


def rule_i_canopy_homogenization(classes, z_rel, x, y, z_raw, spec, provenance, immutable,
                                 wire_idx=None):
    """Rule I: give tree crowns back to vegetation, never touching a utility.

    The wire and pole stages of Rule F read local geometry, and a conifer is full of it: a trunk
    is a vertical line and a branch a horizontal one, so a crown leaves Rule F as red and blue
    speckle around grey holes - and the colour-blind model files much of the rest as Ground or
    Building. This rule undoes that inside the canopy, and only there:

    1. Utility protection. ``immutable`` (``build_immutable_utility_mask``, frozen before this
       rule runs) holds every pole, every confirmed wire and every point inside the 2D wire
       corridor; none of them is a candidate, and the rule fails loudly if one changed anyway.
    2. Crown test. Every other ``source_classes`` point above ``min_hag`` is foliage when its
       ``crown_radius`` neighbourhood holds at least ``min_neighbors`` points and is either
       mostly vegetation already (``min_veg_fraction``) or volumetric - sphericity at least
       ``min_sphericity`` and planarity at most ``max_planarity``, which is what a crown is and
       a cable or a roof is not. ``iterations`` > 1 repeats the vegetation vote so a grey hole
       closes from its green rim inwards; the default is a single pass.
    3. Stratification. Absorbed points take the vegetation_split bands Rule A uses, so a
       recovered point and an AI vegetation point at the same height get the same class.

    Class 14 stays in the default sources on purpose: a wire point that is neither part of a
    confirmed line nor inside its corridor is the crown speckle this rule exists to remove.
    """
    c = spec["canopy_homogenization"]
    if not bool(c["enabled"]):
        print("        [Rule I] Disabled by contract (canopy_homogenization.enabled=false)")
        return {"enabled": False, "absorbed": 0}
    if cKDTree is None:  # pragma: no cover - depends on the environment
        print(f"        [Rule I] [!] scipy is unavailable ({_SCIPY_ERROR}); canopy cleanup skipped")
        return {"enabled": False, "absorbed": 0, "error": str(_SCIPY_ERROR)}

    v = spec["vegetation_split"]
    floor_hag = float(v["spurious_veg_min_hag"])
    lo_max, med_max = float(v["low_veg_max_hag"]), float(v["med_veg_max_hag"])
    conductor_code = int(spec["network_extraction"]["conductor_class"])
    pole_code = int(spec["network_extraction"]["pole_class"])
    sources = tuple(int(s) for s in c["source_classes"])
    radius = float(c["crown_radius"])
    min_n = int(c["min_neighbors"])
    stats = {"enabled": True, "source_classes": list(sources), "absorbed": 0,
             "crown_radius": radius, "corridor_radius": float(c["corridor_radius"]),
             "iterations": int(c["iterations"]),
             "immutable_points": int(np.count_nonzero(immutable))}
    frozen = classes[immutable].copy()

    # The crown neighbourhood is the elevated, non-noise cloud: terrain at HAG ~0 would flatten
    # every crown base into a plane, and noise is not foliage.
    coords = np.column_stack((x, y, z_raw))
    cloud_idx = np.flatnonzero(np.isfinite(coords).all(axis=1) & (z_rel > floor_hag)
                               & ~np.isin(classes, NOISE_CODES))
    if len(cloud_idx) < int(c["crown_k_neighbors"]):
        print(f"        [Rule I] Only {len(cloud_idx):,} elevated pts; canopy cleanup skipped")
        return stats
    lap = _Laps()
    cloud = coords[cloud_idx]
    cloud_cls = classes[cloud_idx]
    cloud_hag = z_rel[cloud_idx]
    tree = _tree(cloud)
    lap("index")

    # --- 1. candidates: the listed classes above min_hag, minus every frozen utility point ---
    in_scope = np.isin(cloud_cls, sources) & (cloud_hag > float(c["min_hag"]))
    cand_pos = np.flatnonzero(in_scope & ~immutable[cloud_idx])
    stats["candidates"] = int(len(cand_pos))
    stats["protected_candidates"] = int(np.count_nonzero(in_scope & immutable[cloud_idx]))
    stats["protected_wire"] = int(np.count_nonzero(
        in_scope & immutable[cloud_idx] & (cloud_cls == conductor_code)))
    print(f"        [Rule I] {stats['candidates']:,} candidates from internal {list(sources)} above "
          f"{float(c['min_hag']):g} m HAG; {stats['protected_candidates']:,} more are frozen by the "
          f"utility mask ({stats['protected_wire']:,} of them wire)")
    if not len(cand_pos):
        return stats

    # --- 2. the crown test: geometry once, the vegetation vote round by round ---
    t0 = time.time()
    n_all = np.asarray(tree.query_ball_point(cloud[cand_pos], r=radius, return_length=True, workers=-1)) - 1
    lap("count")
    plan, sph = _local_shape(cloud, tree, cand_pos, int(c["crown_k_neighbors"]))
    dense = n_all >= min_n
    volumetric = dense & (sph >= float(c["min_sphericity"])) & (plan <= float(c["max_planarity"]))
    lap("shape")
    absorbed = np.zeros(len(cand_pos), dtype=bool)
    by_round = []
    veg_cls = np.isin(cloud_cls, VEG_CODES)
    # The vegetation votes below count vegetation points within a radius of each candidate. The
    # vegetation set only ever grows by the points this rule absorbs, so a count is the count in
    # a base tree built once from the AI's own vegetation plus the count in a small tree of the
    # points absorbed so far - the same integers as rebuilding a tree over the whole set at every
    # vote (five full builds on a 15 M point tile), without the builds.
    base_pos = np.flatnonzero(veg_cls)
    base_tree = _tree(cloud[base_pos]) if len(base_pos) else None

    def veg_count(query_pts, radius_, base, extra_pos):
        n_ = (_ball_count(base, query_pts, radius_) if base is not None
              else np.zeros(len(query_pts), dtype=np.int64))
        if len(extra_pos):
            n_ = n_ + _ball_count(_tree(cloud[extra_pos]), query_pts, radius_)
        return n_

    fresh_pos = None
    for round_no in range(int(c["iterations"])):
        open_ = np.flatnonzero(dense & ~absorbed)
        if fresh_pos is not None:
            # Only a candidate next to a point absorbed last round can have changed its vote.
            if not len(open_):
                break
            _, near = cKDTree(cloud[fresh_pos]).query(cloud[cand_pos[open_]], k=1,
                                                      distance_upper_bound=radius, workers=-1)
            open_ = open_[np.asarray(near) < len(fresh_pos)]
        extra_pos = cand_pos[np.flatnonzero(absorbed)]
        veg_n = (veg_count(cloud[cand_pos[open_]], radius, base_tree, extra_pos)
                 if (base_tree is not None or len(extra_pos)) and len(open_)
                 else np.zeros(len(open_)))
        vote = veg_n / np.maximum(n_all[open_], 1) >= float(c["min_veg_fraction"])
        take = open_[vote | (volumetric[open_] if round_no == 0 else False)]
        by_round.append(int(len(take)))
        if not len(take):
            break
        absorbed[take] = True
        veg_cls[cand_pos[take]] = True
        fresh_pos = cand_pos[take]

    lap("vote")
    # Trunks and branches are lines, not volumes, and their 1.2 m neighbourhood is mostly more
    # trunk - so neither test above can take them, and they reach the deliverable as grey
    # columns and red twigs inside green crowns. Inside a crown the wider neighbourhood is
    # foliage, so a leftover candidate is taken when at least ``interior_veg_fraction`` of the
    # points within ``interior_radius`` are vegetation. It runs on candidates only: confirmed
    # wire, its corridor and every pole are frozen before this rule starts.
    interior = np.zeros(len(cand_pos), dtype=bool)
    if float(c["interior_radius"]) > 0:
        open_ = np.flatnonzero(~absorbed)
        extra_pos = cand_pos[np.flatnonzero(absorbed)]
        if len(open_) and (base_tree is not None or len(extra_pos)):
            r_in = float(c["interior_radius"])
            q = cloud[cand_pos[open_]]
            v_n = veg_count(q, r_in, base_tree, extra_pos)
            a_n = np.asarray(tree.query_ball_point(q, r=r_in, return_length=True, workers=-1)) - 1
            take = open_[(a_n >= min_n)
                         & (v_n / np.maximum(a_n, 1) >= float(c["interior_veg_fraction"]))]
            interior[take] = True
            absorbed[take] = True
    stats["interior"] = int(interior.sum())
    lap("interior")
    # Wire more than ``corridor_radius`` from a confirmed line, measured horizontally, is not a
    # cable: Rule F's linearity test fired on a trunk or a branch. It goes to vegetation by its
    # height band whatever surrounds it - on DU3039046 ~1.8 k of these sit in free air (HAG
    # 9-16 m, nothing green within 3 m), and they go too, by contract. Wire within the radius but
    # deeper than ``corridor_depth`` under the line is not forced: it is a candidate like any
    # other and the crown tests above decide.
    if wire_idx is None:
        wire_idx, _ = _valid_conductor_lines(
            coords, np.flatnonzero(np.isfinite(coords).all(axis=1) & (classes == conductor_code)), c)
    strip = np.zeros(len(cand_pos), dtype=bool)
    open_ = np.flatnonzero(~absorbed & (cloud_cls[cand_pos] == conductor_code))
    if len(open_):
        if len(wire_idx):
            far = np.isinf(cKDTree(coords[wire_idx, :2]).query(
                cloud[cand_pos[open_], :2], k=1, distance_upper_bound=float(c["corridor_radius"]),
                workers=-1)[0])
        else:
            far = np.ones(len(open_), dtype=bool)
        strip[open_[far]] = True
        absorbed[open_[far]] = True
    lap("wire")
    stats["wire_stripped"] = int(strip.sum())
    stats["wire_kept_near_line"] = int(len(open_) - strip.sum())
    print(f"        [Rule I] Wire more than {float(c['corridor_radius']):g} m (2D) from a confirmed "
          f"line: {stats['wire_stripped']:,} pts -> vegetation; "
          f"{stats['wire_kept_near_line']:,} within it but under the corridor depth stay candidates")
    # Morphological closing. Whatever grey or off-line red still sits inside a crown after the
    # tests above is a hole in it: a candidate above ``fill_min_hag`` with ``fill_min_veg`` or more
    # medium / high vegetation points within ``fill_radius`` is filled, ``fill_passes`` times, so
    # the crown closes from its rim inwards. Building is not filled: everything left in that class
    # is a roof Rule L validated (the false ones were re-filed before this rule), and filling it
    # from an adjacent tree would eat ``fill_radius`` metres of roof per pass. The utility mask
    # still holds - confirmed wire, its corridor and poles are never candidates.
    fill_codes = tuple(int(v_) for v_ in c["fill_classes"])
    fill_open = ~absorbed & (cloud_hag[cand_pos] > float(c["fill_min_hag"])) & np.isin(
        cloud_cls[cand_pos], fill_codes)
    filled_per_pass = []
    dense_pos = np.flatnonzero(np.isin(cloud_cls, (4, 5)))     # medium / high vegetation, as filed
    dense_tree = _tree(cloud[dense_pos]) if len(dense_pos) else None
    for _ in range(int(c["fill_passes"])):
        open_ = np.flatnonzero(fill_open & ~absorbed)
        # A candidate that is already medium / high vegetation in the file is in the base tree;
        # counting it again as "absorbed" would double it, so only absorbed non-dense points join.
        extra_pos = cand_pos[np.flatnonzero(absorbed)]
        extra_pos = extra_pos[~np.isin(cloud_cls[extra_pos], (4, 5))]
        if not len(open_) or (dense_tree is None and not len(extra_pos)):
            break
        n_veg = veg_count(cloud[cand_pos[open_]], float(c["fill_radius"]), dense_tree, extra_pos)
        take = open_[n_veg >= int(c["fill_min_veg"])]
        filled_per_pass.append(int(len(take)))
        if not len(take):
            break
        absorbed[take] = True
    stats["filled_per_pass"] = filled_per_pass
    lap("fill")
    # Volumetric scatter at a wider scale: a red / grey cluster inside a crown can be wider than the
    # 1.2-2 m neighbourhoods above, so its own points are all that the small tests see. Over a
    # ``scatter_radius`` sphere it is part of the crown again: scatter = l3 / (l1 + l2 + l3) above
    # ``min_scatter`` is a volume, where a cable stays near 0. Building is left out for the same
    # reason as in the fill - what is left of it is a validated roof, and a 2.5 m sphere on a roof
    # corner spans two walls and the roof, which reads as a volume.
    sc_codes = tuple(int(v_) for v_ in c["fill_classes"])
    sc_open = np.flatnonzero(~absorbed & (cloud_hag[cand_pos] > float(c["min_hag"]))
                             & np.isin(cloud_cls[cand_pos], sc_codes))
    scatter_taken = 0
    if len(sc_open):
        r_sc = float(c["scatter_radius"])
        hits = tree.query_ball_point(cloud[cand_pos[sc_open]], r=r_sc)
        take = np.zeros(len(sc_open), dtype=bool)
        for j, nb in enumerate(hits):
            if len(nb) < int(c["scatter_min_points"]):
                continue
            q = cloud[nb]
            q = q - q.mean(axis=0)
            ev = np.linalg.eigvalsh(q.T @ q / len(nb))
            total = float(ev.sum())
            take[j] = total > 0 and float(ev[0]) / total > float(c["min_scatter"])
        absorbed[sc_open[take]] = True
        scatter_taken = int(take.sum())
    stats["scatter_absorbed"] = scatter_taken
    lap("scatter")
    print(f"        [Rule I] Volumetric scatter (l3 / sum > {float(c['min_scatter']):g} over "
          f"{float(c['scatter_radius']):g} m, internal {list(sc_codes)}): {scatter_taken:,} pts")
    print(f"        [Rule I] Crown hole fill (internal {list(fill_codes)} above "
          f"{float(c['fill_min_hag']):g} m with >= {int(c['fill_min_veg'])} medium / high vegetation "
          f"pts within {float(c['fill_radius']):g} m): filled per pass "
          f"{', '.join(f'{n:,}' for n in filled_per_pass) or '0'}")
    print(f"        [Rule I] Canopy interior (veg fraction >= {float(c['interior_veg_fraction']):g} "
          f"within {float(c['interior_radius']):g} m): {stats['interior']:,} trunk / branch pts")
    stats["volumetric"] = int(volumetric.sum())
    stats["per_round"] = by_round
    stats["seconds"] = round(time.time() - t0, 2)
    stats["stage_seconds"] = dict(lap.seconds)
    print(f"        [Rule I] Crown test (>= {min_n} pts within {radius:g} m, and veg fraction >= "
          f"{float(c['min_veg_fraction']):g} or sphericity >= {float(c['min_sphericity']):g} with "
          f"planarity <= {float(c['max_planarity']):g}, k={int(c['crown_k_neighbors'])}): "
          f"{stats['volumetric']:,} volumetric; absorbed per round "
          f"{', '.join(f'{n:,}' for n in by_round)} in {stats['seconds']:.2f} s")

    # --- 3. stratify what was absorbed with Rule A's bands ---
    picked = cloud_idx[cand_pos[absorbed]]
    before = classes[picked].copy()
    hag = z_rel[picked]
    target = np.where(hag <= lo_max, 3, np.where(hag <= med_max, 4, 5)).astype(classes.dtype)
    classes[picked] = target
    provenance[picked] = ORIGIN_CANOPY
    if not np.array_equal(classes[immutable], frozen):   # pragma: no cover - defensive
        raise SystemExit("[X] Rule I modified a frozen utility point (pole / wire / corridor)")
    stats["absorbed"] = int(len(picked))
    stats["absorbed_by_source"] = {int(s): int(np.count_nonzero(before == s)) for s in sources}
    stats["absorbed_by_target"] = {int(t): int(np.count_nonzero(target == t)) for t in (3, 4, 5)}
    print(f"        [Rule I] Absorbed {stats['absorbed']:,} pts into vegetation from: " + " | ".join(
        f"class {s} {class_name(s)}: {n:,}" for s, n in stats["absorbed_by_source"].items() if n))
    print(f"                  stratified {floor_hag:g}-{lo_max:g}-{med_max:g} m -> " + ", ".join(
        f"{t}: {n:,} (delivered as {_delivered_label(spec, t)})"
        for t, n in stats["absorbed_by_target"].items()))

    # Evidence for the pole rules: Rule I itself never touches a pole.
    pole_pos = np.flatnonzero((cloud_cls == pole_code) & (cloud_hag > float(c["min_hag"])))
    if len(pole_pos):
        p_n = np.asarray(tree.query_ball_point(cloud[pole_pos], r=radius,
                                               return_length=True, workers=-1)) - 1
        p_plan, p_sph = _local_shape(cloud, tree, pole_pos, int(c["crown_k_neighbors"]))
        crowned = ((p_n >= min_n) & (p_sph >= float(c["min_sphericity"]))
                   & (p_plan <= float(c["max_planarity"])))
        stats["pole_points_in_canopy"] = int(crowned.sum())
        print(f"                  {stats['pole_points_in_canopy']:,} of {len(pole_pos):,} "
              f"internal {pole_code} pole pts sit in a volumetric crown; frozen as poles - "
              "only the pole rules (pole_validation) may revert them")
    lap("finish")
    print(f"        [Rule I] time by stage: {lap.line()}")
    return stats


def rule_j_exposed_pole_truncation(classes, z_rel, x, y, z_raw, spec, provenance,
                                   wire_idx=None, traced_poles=None):
    """Rule J: a pole inside vegetation is only a pole where it stands clear of the canopy.

    Rule F's column flood claims everything inside a ``pole_cylinder.radius`` cylinder from
    ``base_hag`` to the top seed, so a pole standing in a hedge or a tree row - and a conifer
    trunk that Rule F mistook for one - leaves Rule F as a solid blue column through the foliage.
    This rule runs after Rule I, so the canopy around each column is already vegetation:

    1. Columns. Pole points are linked in 2D (``column_link``) into vertical columns.
    2. Envelope. ``Z_veg_max`` is the ``veg_height_percentile`` (100 = max) of the vegetation
       within ``veg_radius`` (2D) of the column centroid; ``Z_cutoff = Z_veg_max + cutoff_margin``.
       A column whose base is already above the cutoff is not embedded and is left untouched.
    3. Exposed section. In an embedded column only the points above ``Z_cutoff`` can stay pole,
       and only when the column holds a real wire: its head (top ``head_depth``) lies within
       ``attach_radius`` of a *confirmed* conductor (``_valid_conductor_lines``, the same test
       Rule I protects) and the exposed section is a vertical line (``min_exposed_points``,
       ``min_exposed_linearity``). That wire attachment is what replaces ground attachment: the
       exposed section of an embedded pole never reaches the ground, and a vertical stub above
       the canopy that holds no wire is a tree top, not a pole.
    4. Everything else in an embedded column becomes vegetation, stratified with Rule A's bands.

    Measured on the 123 tile: the two open poles stand 12.6-12.9 m clear of the canopy and keep
    89-95 % of their points; six 5 m "columns" inside the tree rows top out exactly at the canopy
    (top minus Z_veg_max -0.4..+0.3 m) and their only wire contact is in-crown speckle, so they
    go back to vegetation. ``pole_cylinder.radius`` is 0.45 m, so a flooded column measures
    r90 ~0.43 m whatever it is: a 0.35 m thinness gate would test the flood, not the pole, and is
    not applied (see also the pole_validation note on thinness).
    """
    j = spec["pole_truncation"]
    if not bool(j["enabled"]):
        print("        [Rule J] Disabled by contract (pole_truncation.enabled=false)")
        return {"enabled": False, "truncated_points": 0}
    if cKDTree is None:  # pragma: no cover - depends on the environment
        print(f"        [Rule J] [!] scipy is unavailable ({_SCIPY_ERROR}); pole truncation skipped")
        return {"enabled": False, "truncated_points": 0, "error": str(_SCIPY_ERROR)}
    pole_code = int(spec["network_extraction"]["pole_class"])
    conductor_code = int(spec["network_extraction"]["conductor_class"])
    v = spec["vegetation_split"]
    lo_max, med_max = float(v["low_veg_max_hag"]), float(v["med_veg_max_hag"])
    stats = {"enabled": True, "columns": 0, "embedded": 0, "open_shaft": 0, "kept_exposed": 0,
             "submerged": 0, "unattached": 0, "not_linear": 0, "untouched": 0,
             "pole_points_before": 0, "kept_points": 0, "truncated_points": 0}

    lap = _Laps()
    coords = np.column_stack((x, y, z_raw))
    finite = np.isfinite(coords).all(axis=1)
    pole_idx = np.flatnonzero(finite & (classes == pole_code))
    stats["pole_points_before"] = int(len(pole_idx))
    veg_idx = np.flatnonzero(finite & np.isin(classes, VEG_CODES))
    if not len(pole_idx) or not len(veg_idx):
        print(f"        [Rule J] {len(pole_idx):,} pole pts, {len(veg_idx):,} vegetation pts: "
              "nothing to truncate")
        return stats

    # --- 1. columns: pole points linked in 2D, on a fine XY grid so the link graph stays small ---
    cell_key = np.floor(coords[pole_idx, :2] / float(j["column_cell"])).astype(np.int64)
    _, first, inverse = np.unique(cell_key, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    nodes = coords[pole_idx[first], :2]
    links = cKDTree(nodes).query_pairs(r=float(j["column_link"]), output_type="ndarray")
    column = _link_labels(len(nodes), links)[inverse]
    counts = np.bincount(column)
    order = np.argsort(column, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    stats["columns"] = int(np.count_nonzero(counts))
    lap("columns")

    # --- confirmed wire, the same lines Rule I protects ---
    if wire_idx is None:
        wire_idx, _ = _valid_conductor_lines(
            coords, np.flatnonzero(finite & (classes == conductor_code)),
            spec["canopy_homogenization"])
    wire_tree = _tree(coords[wire_idx]) if len(wire_idx) else None
    stats["confirmed_wire_points"] = int(len(wire_idx))

    lap("wire")
    # --- 2. everything below is per column, done as array reductions over the column-sorted pole
    # points (one pass each) instead of a Python loop over what is, on a dense tile, ~100 k columns.
    veg_xy_tree = _tree(coords[veg_idx, :2])
    veg_z = coords[veg_idx, 2]
    lap("veg index")
    pct = float(j["veg_height_percentile"])
    margin = float(j["cutoff_margin"])
    n_col = len(counts)
    pole_sorted = pole_idx[order]
    pts = coords[pole_sorted]                      # all pole points, grouped by column
    col_of = column[order]
    z_pt = pts[:, 2]

    # Envelope: around the column's whole footprint, not just its centroid - a pole clump Rule F
    # made out of a crown can be metres wide, and its centroid then sees only more pole.
    cell_w = float(j["column_cell"])
    fkey = np.column_stack((col_of, np.floor(pts[:, :2] / cell_w))).astype(np.int64)
    fuq = np.unique(fkey, axis=0)                  # distinct (column, footprint cell), by column
    f_col = fuq[:, 0]
    f_xy = (fuq[:, 1:].astype(np.float64) + 0.5) * cell_w
    lap("footprint")
    r_veg = float(j["veg_radius"])
    env = np.full(n_col, -np.inf)
    has_veg = np.zeros(n_col, dtype=bool)
    z_min = np.minimum.reduceat(z_pt, starts)
    z_max = np.maximum.reduceat(z_pt, starts)
    min_exposed = int(j["min_exposed_points"])
    if pct >= 100.0:
        # The 100th percentile is the maximum, which distributes over the footprint cells: the
        # column's envelope is the highest of its cells' highest vegetation point.
        def accumulate(rows):
            for lo in range(0, len(rows), 20_000):
                sel_ = rows[lo:lo + 20_000]
                hits = veg_xy_tree.query_ball_point(f_xy[sel_], r=r_veg, workers=-1)
                lens = np.fromiter(map(len, hits), dtype=np.int64, count=len(hits))
                total = int(lens.sum())
                if not total:
                    continue
                flat = np.fromiter(chain.from_iterable(hits), dtype=np.int64, count=total)
                nz = np.flatnonzero(lens)
                seg = np.concatenate(([0], np.cumsum(lens)[:-1]))[nz]
                cell_max = np.maximum.reduceat(veg_z[flat], seg)
                cols_hit = f_col[sel_][nz]
                np.maximum.at(env, cols_hit, cell_max)
                has_veg[cols_hit] = True

        # A column is decided by two things only: is the canopy above its lowest point
        # (embedded), and do fewer than ``min_exposed_points`` of its points stand above the
        # cutoff (submerged)? Both are monotone in the envelope, so a *lower bound* on it settles
        # them whenever they already hold at the bound. Every ``stride``-th footprint cell is a
        # subset of the cells, and a maximum over a subset is a lower bound; on a dense tile that
        # proves nearly every column (1,720 of 1,733 on CAIOLO) at a sixteenth of the vegetation
        # pairs. Only the columns it cannot prove are recomputed from all their cells, so the
        # decisions - and everything derived from them - are exactly those of the full pass.
        stride = 16
        accumulate(np.arange(0, len(f_xy), stride))
        cut_lb = env + margin
        above_lb = np.add.reduceat((z_pt > cut_lb[col_of]).astype(np.int64), starts)
        proven = has_veg & (cut_lb > z_min) & (above_lb < min_exposed)
        redo = ~proven
        env[redo] = -np.inf
        has_veg[redo] = False
        accumulate(np.flatnonzero(redo[f_col]))
        stats["envelope_exact_columns"] = int(np.count_nonzero(redo))
    else:
        # A lower percentile is taken over the union of distinct vegetation points, so the
        # footprint cells cannot be reduced independently: one union per column, as ever.
        f_start = np.searchsorted(f_col, np.arange(n_col), side="left")
        f_stop = np.searchsorted(f_col, np.arange(n_col), side="right")
        for col in np.flatnonzero(f_stop > f_start):
            near = set()
            for hits in veg_xy_tree.query_ball_point(f_xy[f_start[col]:f_stop[col]], r=r_veg):
                near.update(hits)
            if near:
                env[col] = float(np.percentile(veg_z[list(near)], pct))
                has_veg[col] = True
    cutoff = env + margin
    lap("envelope")
    embedded = has_veg & (cutoff > z_min)          # the canopy reaches the column: it is embedded
    # ...but only when it really surrounds the shaft. A pole standing in the open can have a few
    # stray vegetation returns at head height (cross-arm and insulator speckle that Rule I filed
    # as canopy); the maximum of those alone said "the canopy reaches the top" and sent the whole
    # open pole to vegetation. Vegetation has to be present on the lower half of the shaft too.
    shaft_min = int(j.get("shaft_min_veg", 0))
    if shaft_min > 0 and embedded.any():
        z_mid = 0.5 * (z_min + z_max)
        shaft_veg = np.zeros(n_col, dtype=np.int64)
        rows_all = np.flatnonzero(embedded[f_col])
        for lo in range(0, len(rows_all), 20_000):
            rows = rows_all[lo:lo + 20_000]
            hits = veg_xy_tree.query_ball_point(f_xy[rows], r=r_veg, workers=-1)
            lens = np.fromiter(map(len, hits), dtype=np.int64, count=len(hits))
            total = int(lens.sum())
            if not total:
                continue
            flat = np.fromiter(chain.from_iterable(hits), dtype=np.int64, count=total)
            owner = np.repeat(f_col[rows], lens)
            in_shaft = (veg_z[flat] >= z_min[owner] + 0.5) & (veg_z[flat] <= z_mid[owner])
            shaft_veg += np.bincount(owner[in_shaft], minlength=n_col)
        # Only a real column can be an open pole: a handful of stray points or a short stub in
        # a crown stays under the canopy test (it has no shaft to be open).
        col_n = counts.astype(np.int64)
        real = (col_n >= int(j.get("shaft_min_points", 30)))             & ((z_max - z_min) >= float(j.get("shaft_min_height", 4.0)))
        open_shaft = embedded & real & (shaft_veg < shaft_min)
        stats["open_shaft"] = int(np.count_nonzero(open_shaft))
        embedded &= ~open_shaft
    stats["untouched"] = int(n_col - np.count_nonzero(embedded))
    stats["embedded"] = int(np.count_nonzero(embedded))
    exposed_pt = z_pt > cutoff[col_of]
    n_exposed = np.add.reduceat(exposed_pt.astype(np.int64), starts)
    submerged = embedded & (n_exposed < min_exposed)
    stats["submerged"] = int(np.count_nonzero(submerged))
    testable = embedded & ~submerged

    # Wire attachment: any point in the column's top ``head_depth`` within ``attach_radius`` of a
    # confirmed conductor.
    attached = np.zeros(n_col, dtype=bool)
    head_pt = (z_pt >= (z_max - float(j["head_depth"]))[col_of]) & testable[col_of]
    if wire_tree is not None and head_pt.any():
        d = wire_tree.query(pts[head_pt], k=1, distance_upper_bound=float(j["attach_radius"]),
                            workers=-1)[0]
        attached[np.unique(col_of[head_pt][np.asarray(d) < np.inf])] = True
    stats["unattached"] = int(np.count_nonzero(testable & ~attached))
    lin_test = testable & attached

    # Is the exposed section a vertical line? Covariance of each tested column's exposed points.
    keep_col = np.zeros(n_col, dtype=bool)
    e_pt = exposed_pt & lin_test[col_of]
    if e_pt.any():
        ec = col_of[e_pt]
        ep = pts[e_pt]
        n_e = np.bincount(ec, minlength=n_col).astype(np.float64)
        tested = np.flatnonzero(lin_test)
        mean = np.column_stack([np.bincount(ec, weights=ep[:, d_], minlength=n_col)
                                for d_ in range(3)])[tested] / n_e[tested, None]
        mean_full = np.zeros((n_col, 3))
        mean_full[tested] = mean
        cen = ep - mean_full[ec]
        cov = np.zeros((len(tested), 3, 3))
        pos = np.full(n_col, -1, dtype=np.int64)
        pos[tested] = np.arange(len(tested))
        for a_ in range(3):
            for b_ in range(a_, 3):
                v_ = np.bincount(pos[ec], weights=cen[:, a_] * cen[:, b_], minlength=len(tested))
                cov[:, a_, b_] = cov[:, b_, a_] = v_ / n_e[tested]
        evals, evecs = np.linalg.eigh(cov)
        l1 = evals[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            linear = ((l1 > 1e-12) & ((l1 - evals[:, 1]) / l1 >= float(j["min_exposed_linearity"]))
                      & (np.abs(evecs[:, 2, 2]) >= float(j["min_exposed_verticality"])))
        keep_col[tested[linear]] = True
        stats["not_linear"] = int(np.count_nonzero(~linear))
    stats["kept_exposed"] = int(np.count_nonzero(keep_col))
    # whole column to vegetation, or - for a kept column - everything below its exposed top
    take_pt = embedded[col_of] & ~(keep_col[col_of] & exposed_pt)
    if traced_poles is not None and len(traced_poles):
        # a pole the network trace reached along its wire is a pole, canopy or not
        traced_mask = np.zeros(len(classes), dtype=bool)
        traced_mask[np.asarray(traced_poles, dtype=np.int64)] = True
        take_pt &= ~traced_mask[pole_sorted]
    picked = pole_sorted[take_pt]
    lap("tests")
    hag = z_rel[picked]
    target = np.where(hag <= lo_max, 3, np.where(hag <= med_max, 4, 5)).astype(classes.dtype)
    classes[picked] = target
    provenance[picked] = ORIGIN_POLE_TRUNCATION
    stats["truncated_points"] = int(len(picked))
    stats["kept_points"] = int(np.count_nonzero(classes[pole_idx] == pole_code))
    stats["truncated_by_target"] = {int(t): int(np.count_nonzero(target == t)) for t in (3, 4, 5)}
    print(f"        [Rule J] Pole columns vs the vegetation envelope (p{pct:g} of vegetation Z "
          f"within {float(j['veg_radius']):g} m + {margin:g} m): {stats['columns']:,} columns, "
          f"{stats['embedded']:,} embedded, {stats['untouched']:,} clear of the canopy")
    print(f"                  embedded: {stats['kept_exposed']:,} keep their exposed top "
          f"(head within {float(j['attach_radius']):g} m of one of "
          f"{stats['confirmed_wire_points']:,} confirmed wire pts, linear section); "
          f"{stats['submerged']:,} fully submerged, {stats['unattached']:,} hold no confirmed "
          f"wire, {stats['not_linear']:,} not a vertical line -> whole column to vegetation")
    print(f"                  {stats['truncated_points']:,} of {stats['pole_points_before']:,} pole "
          f"pts -> vegetation (" + ", ".join(
              f"{t}: {n:,}" for t, n in stats["truncated_by_target"].items())
          + f"); {stats['kept_points']:,} stay internal {pole_code}, delivered as "
          f"{_delivered_label(spec, pole_code)}")
    lap("finish")
    print(f"                  time by stage: {lap.line()}")
    return stats


def rule_k_wire_guided_pole_recovery(classes, z_rel, x, y, z_raw, spec, provenance,
                                     wire_idx=None):
    """Rule K: find the poles Rule F missed by looking down from where the wire is held.

    Rule F finds poles bottom-up, from the verticality of each point's neighbourhood, and a pole
    that is sparsely sampled or stands at the edge of a tree row never produces enough seeds: it
    reaches the deliverable as a grey column under the wire. A wire, though, says where its
    supports are - it is highest where it is held, and sags in between - so this rule works
    top-down:

    1. Support nodes. Confirmed wire (``_valid_conductor_lines``) is gridded on ``node_cell``;
       a cell whose top is within ``node_tolerance`` of the highest wire within ``node_radius`` is
       a local height maximum - a support, a bend, or a convergence of spans. Maxima closer than
       ``node_merge`` are one node, represented by its highest cell.
    2. Downward column. Under each node with no pole yet within ``search_radius``, every
       ``axis_cell`` sub-cell is tried as a pole axis - the pole stands under the middle of its
       cross-arm, not under each phase - and the points of ``source_classes`` within
       ``column_radius`` of it are walked down from the wire. The run starts within ``top_gap``
       of the wire and ends at the first vertical gap over ``max_gap``; the longest run wins, and
       it is a pole when it spans more than ``min_column_span``.
    3. Vegetation cutoff, shared with Rule J (``pole_truncation``): only the run above
       ``Z_veg_max + cutoff_margin`` becomes pole, where ``Z_veg_max`` is measured on the
       vegetation around the column (excluding the column itself). The exposed part must hold
       ``min_exposed_points`` and stand ``min_exposed_span`` clear of the canopy (a 1 m stub on
       a crown under in-forest wire was the one false recovery on DU3039046), and the ring around it (to ``search_radius``, wire excluded) must
       not hold more than ``max_ring_ratio`` times as many points above the cutoff - a crown that
       reaches up to the wire fills that ring, a pole does not.

    Measured on the 123 tile: the grey pole under the bend at x 256312 runs 14.1 m down from the
    wire, the canopy around it is cut at 299.7 m, and 151 points above it become pole against 2
    ring points; every other node is either an existing pole or open air under the span.
    """
    k = spec["pole_recovery"]
    if not bool(k["enabled"]):
        print("        [Rule K] Disabled by contract (pole_recovery.enabled=false)")
        return {"enabled": False, "recovered_points": 0}
    if cKDTree is None:  # pragma: no cover - depends on the environment
        print(f"        [Rule K] [!] scipy is unavailable ({_SCIPY_ERROR}); pole recovery skipped")
        return {"enabled": False, "recovered_points": 0, "error": str(_SCIPY_ERROR)}
    pole_code = int(spec["network_extraction"]["pole_class"])
    conductor_code = int(spec["network_extraction"]["conductor_class"])
    trunc = spec["pole_truncation"]
    stats = {"enabled": True, "nodes": 0, "near_existing_pole": 0, "no_column": 0,
             "too_short": 0, "submerged": 0, "crowded": 0, "recovered_poles": 0,
             "recovered_points": 0}

    coords = np.column_stack((x, y, z_raw))
    finite = np.isfinite(coords).all(axis=1)
    if wire_idx is None:
        wire_idx, _ = _valid_conductor_lines(
            coords, np.flatnonzero(finite & (classes == conductor_code)),
            spec["canopy_homogenization"])
    stats["confirmed_wire_points"] = int(len(wire_idx))
    if not len(wire_idx):
        print("        [Rule K] No confirmed wire: no support nodes to search under")
        return stats

    # --- 1. support nodes: local height maxima of the confirmed wire ---
    wire = coords[wire_idx]
    cells, inverse = np.unique(np.floor(wire[:, :2] / float(k["node_cell"])).astype(np.int64),
                               axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    top = np.full(len(cells), -np.inf)
    np.maximum.at(top, inverse, wire[:, 2])
    centre = (cells + 0.5) * float(k["node_cell"])
    around = cKDTree(centre).query_ball_point(centre, r=float(k["node_radius"]))
    local_max = np.array([top[i] >= top[nb].max() - float(k["node_tolerance"])
                          for i, nb in enumerate(around)], dtype=bool)
    peak = np.flatnonzero(local_max)
    links = cKDTree(centre[peak]).query_pairs(r=float(k["node_merge"]), output_type="ndarray")
    group = _link_labels(len(peak), links)
    nodes = []
    for g in np.unique(group):
        members = peak[group == g]
        best = members[np.argmax(top[members])]
        nodes.append((centre[best], float(top[best])))
    stats["nodes"] = len(nodes)

    # --- 2./3. downward column under each node, cut at the vegetation envelope ---
    sources = np.isin(classes, tuple(int(s) for s in k["source_classes"])) & finite
    is_veg = np.isin(classes, VEG_CODES)
    is_pole = classes == pole_code
    is_wire = classes == conductor_code
    flat_tree = _tree(coords[:, :2])
    search_r = float(k["search_radius"])
    col_r = float(k["column_radius"])
    axis_cell = float(k["axis_cell"])
    recovered = []
    for xy, wire_z in nodes:
        near = np.asarray(flat_tree.query_ball_point(xy, r=search_r), dtype=np.int64)
        if np.any(is_pole[near]):
            stats["near_existing_pole"] += 1
            continue
        cand = near[sources[near] & (coords[near, 2] <= wire_z + float(k["top_margin"]))]
        best = None
        for axis in np.unique(np.floor(coords[cand, :2] / axis_cell), axis=0):
            a = (axis + 0.5) * axis_cell
            members = cand[np.hypot(coords[cand, 0] - a[0], coords[cand, 1] - a[1]) <= col_r]
            zs = np.sort(coords[members, 2])[::-1]
            if not len(zs) or zs[0] < wire_z - float(k["top_gap"]):
                continue
            gaps = np.flatnonzero(-np.diff(zs) > float(k["max_gap"]))
            bottom = float(zs[gaps[0]] if len(gaps) else zs[-1])
            run = float(zs[0]) - bottom
            if best is None or run > best[0]:
                best = (run, a, members, bottom)
        if best is None:
            stats["no_column"] += 1
            continue
        run, a, members, bottom = best
        if run <= float(k["min_column_span"]):
            stats["too_short"] += 1
            continue
        dist = np.hypot(coords[near, 0] - a[0], coords[near, 1] - a[1])
        ring = near[dist > col_r]
        veg_z = coords[ring[is_veg[ring]], 2]
        cutoff = (float(np.percentile(veg_z, float(trunc["veg_height_percentile"])))
                  + float(trunc["cutoff_margin"])) if len(veg_z) else -np.inf
        exposed = members[(coords[members, 2] >= bottom) & (coords[members, 2] > cutoff)]
        if (len(exposed) < int(k["min_exposed_points"])
                or float(np.ptp(coords[exposed, 2])) < float(k["min_exposed_span"])):
            stats["submerged"] += 1
            continue
        crowd = int(np.count_nonzero((coords[ring, 2] > cutoff) & ~is_wire[ring]
                                     & (coords[ring, 2] <= wire_z + float(k["top_margin"]))))
        if crowd > float(k["max_ring_ratio"]) * len(exposed):
            stats["crowded"] += 1
            continue
        stats["recovered_poles"] += 1
        recovered.append(exposed)

    picked = np.unique(np.concatenate(recovered)) if recovered else np.zeros(0, dtype=np.int64)
    stats["recovered_by_source"] = {int(cc): int(np.count_nonzero(classes[picked] == cc))
                                    for cc in np.unique(classes[picked])}
    classes[picked] = pole_code
    provenance[picked] = ORIGIN_POLE_RECOVERY
    stats["recovered_points"] = int(len(picked))
    print(f"        [Rule K] Wire support nodes (confirmed wire within {float(k['node_tolerance']):g} m "
          f"of its highest point inside {float(k['node_radius']):g} m): {stats['nodes']:,} nodes, "
          f"{stats['near_existing_pole']:,} already on a pole")
    print(f"                  downward column (R {col_r:g} m, gap <= {float(k['max_gap']):g} m, span "
          f"> {float(k['min_column_span']):g} m): {stats['recovered_poles']:,} poles recovered; "
          f"{stats['no_column']:,} open air, {stats['too_short']:,} too short, "
          f"{stats['submerged']:,} submerged in the canopy (< {float(k['min_exposed_span']):g} m or < {int(k['min_exposed_points'])} pts clear of it), {stats['crowded']:,} inside a crown")
    print(f"                  {stats['recovered_points']:,} pts -> internal {pole_code} "
          f"({_delivered_label(spec, pole_code)}) from: " + (" | ".join(
              f"class {cc} {class_name(cc)}: {n:,}"
              for cc, n in stats["recovered_by_source"].items()) or "-"))
    return stats


def rule_l_building_validation(classes, z_rel, x, y, z_raw, spec, provenance):
    """Rule L: a building is a roof - pink on the ground or in a crown is not.

    The colour-blind model files sloped ground patches and parts of crowns as Building, and on
    DU3039046 that is 97 % of the delivered pink (17,434 of 17,879 pts), 7,804 of it below 2 m
    HAG. So every Building point is grouped into clusters (``cluster_cell`` grid, linked within
    ``cluster_link`` in 3D) and a cluster stays Building only when its roof - the points at or
    strictly above ``min_roof_hag`` - exists and is a building's roof:
      * roof normal close to vertical (|Nz| > ``min_normal_z``; a 30 deg pitch is 0.87), and
      * roof footprint at least ``min_roof_area`` m2 on a ``footprint_cell`` grid.
    The test is on the roof and not on the whole cluster's lowest point: a real building's walls
    reach the ground, so "minimum HAG >= 2 m" over the cluster would delete every building.
    Nothing below the roof height is kept, walls included (by contract), and a roof point whose
    ``surround_radius`` neighbourhood is at least ``max_surround_fraction`` ground and vegetation
    is dropped too: that is pink speckle in the canopy or on the slope.

    Every point that is not kept is re-filed by height with Rule A's bands: at or below the
    spurious threshold it is Ground; up to ``min_roof_hag`` it is vegetation (shrub / slope
    clutter); above that it becomes Unclassified, so Rule I's crown test - which runs next -
    decides between vegetation and grey on geometry.
    """
    b = spec["building_validation"]
    if not bool(b["enabled"]):
        print("        [Rule L] Disabled by contract (building_validation.enabled=false)")
        return {"enabled": False, "reverted_points": 0}
    if cKDTree is None:  # pragma: no cover - depends on the environment
        print(f"        [Rule L] [!] scipy is unavailable ({_SCIPY_ERROR}); building validation skipped")
        return {"enabled": False, "reverted_points": 0, "error": str(_SCIPY_ERROR)}
    building_code = 6
    v = spec["vegetation_split"]
    spur, lo_max, med_max = (float(v["spurious_veg_min_hag"]), float(v["low_veg_max_hag"]),
                             float(v["med_veg_max_hag"]))
    roof_hag = float(b["min_roof_hag"])
    stats = {"enabled": True, "building_points": 0, "clusters": 0, "kept_clusters": 0,
             "no_roof": 0, "roof_not_flat": 0, "roof_too_small": 0, "kept_points": 0,
             "reverted_points": 0}
    coords = np.column_stack((x, y, z_raw))
    idx = np.flatnonzero(np.isfinite(coords).all(axis=1) & (classes == building_code))
    stats["building_points"] = int(len(idx))
    if not len(idx):
        print("        [Rule L] No Building points to validate")
        return stats

    pts = coords[idx]
    keys = np.floor(pts / float(b["cluster_cell"])).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    nodes = pts[first]
    links = cKDTree(nodes).query_pairs(r=float(b["cluster_link"]), output_type="ndarray")
    cluster = _link_labels(len(nodes), links)[inverse]
    counts = np.bincount(cluster)
    order = np.argsort(cluster, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    stats["clusters"] = int(np.count_nonzero(counts))

    keep = np.zeros(len(idx), dtype=bool)
    cell = float(b["footprint_cell"])
    for g in np.flatnonzero(counts):
        pos = order[starts[g]:starts[g] + counts[g]]
        hag = z_rel[idx[pos]]
        roof = pos[hag > roof_hag]
        if len(roof) < 3:
            stats["no_roof"] += 1
            continue
        centred = pts[roof] - pts[roof].mean(axis=0)
        evecs = np.linalg.eigh(centred.T @ centred / len(roof))[1]
        if abs(float(evecs[2, 0])) <= float(b["min_normal_z"]):
            stats["roof_not_flat"] += 1
            continue
        roof_cells = np.unique(np.floor(pts[roof, :2] / cell).astype(np.int64), axis=0)
        if len(roof_cells) * cell * cell < float(b["min_roof_area"]):
            stats["roof_too_small"] += 1
            continue
        stats["kept_clusters"] += 1
        # Only the roof is kept: by contract nothing below min_roof_hag is delivered as Building,
        # walls included - on a slope the lower walls and the ground bleed are the same pink.
        keep[roof] = True

    # A roof point whose own neighbourhood is mostly ground and vegetation is pink speckle inside
    # the canopy or on the slope, not a roof, whatever cluster it was linked into.
    kept_idx = idx[keep]
    if len(kept_idx):
        r_s = float(b["surround_radius"])
        nat = np.flatnonzero(np.isin(classes, (2,) + VEG_CODES))
        all_n = np.asarray(_tree(coords[np.isfinite(coords).all(axis=1)]).query_ball_point(
            coords[kept_idx], r=r_s, return_length=True, workers=-1)) - 1
        nat_n = (np.asarray(_tree(coords[nat]).query_ball_point(
            coords[kept_idx], r=r_s, return_length=True, workers=-1)) if len(nat) else np.zeros(len(kept_idx)))
        buried = nat_n / np.maximum(all_n, 1) >= float(b["max_surround_fraction"])
        keep[np.flatnonzero(keep)[buried]] = False
        stats["surrounded"] = int(buried.sum())

    # A cluster that passed on one flat patch also carries every other point above the roof line
    # that the model linked to it - on CEDRASCO a tree row beside a roof, where 72 % of the pink
    # that survived was crown-shaped and 9 % flat. Inside a validated cluster only roof-shaped
    # points stay: a ``shape_k``-point neighbourhood that is not volumetric (sphericity above
    # ``max_roof_sphericity``). The rim a real roof loses here (edges and eaves are not planar) is
    # given back by Rule R's eave stage, which grows each instance by a metre or so, not by a hull.
    kept_idx = idx[keep]
    stats["volumetric_dropped"] = 0
    if len(kept_idx):
        fin = np.flatnonzero(np.isfinite(coords).all(axis=1) & (z_rel > 0.15))
        shape_tree = _tree(coords[fin])
        pos = np.searchsorted(fin, kept_idx)
        _, sph = _local_shape(coords[fin], shape_tree, pos, int(b["shape_k"]))
        volumetric = sph > float(b["max_roof_sphericity"])
        keep[np.flatnonzero(keep)[volumetric]] = False
        stats["volumetric_dropped"] = int(volumetric.sum())

    out = idx[~keep]
    hag = z_rel[out]
    target = np.where(hag <= spur, 2,
                      np.where(hag <= lo_max, 3,
                               np.where(hag < roof_hag, np.where(hag <= med_max, 4, 5), 1)))
    target = target.astype(classes.dtype)
    classes[out] = target
    provenance[out] = ORIGIN_BUILDING_VALIDATION
    stats["kept_points"] = int(keep.sum())
    stats["reverted_points"] = int(len(out))
    stats["reverted_by_target"] = {int(t): int(np.count_nonzero(target == t))
                                   for t in np.unique(target)}
    print(f"        [Rule L] Building validation (roof >= {roof_hag:g} m HAG, |Nz| > "
          f"{float(b['min_normal_z']):g}, footprint >= {float(b['min_roof_area']):g} m2): "
          f"{stats['clusters']:,} clusters, {stats['kept_clusters']:,} kept; "
          f"{stats['no_roof']:,} have no roof, {stats['roof_not_flat']:,} not flat, "
          f"{stats['roof_too_small']:,} too small; {stats.get('surrounded', 0):,} roof pts "
          f"dropped as surrounded by ground / vegetation (>= {float(b['max_surround_fraction']):g} "
          f"within {float(b['surround_radius']):g} m); {stats['volumetric_dropped']:,} crown-shaped "
          f"pts dropped (sphericity > {float(b['max_roof_sphericity']):g})")
    print(f"                  {stats['kept_points']:,} pts stay Building; {stats['reverted_points']:,} "
          f"re-filed: " + " | ".join(f"internal {t} {class_name(t)}: {n:,}"
                                     for t, n in stats["reverted_by_target"].items()))
    return stats


def rule_n_bridge_disposal(classes, z_rel, x, y, spec, provenance):
    """Rule N: the deliverable has no Bridge class, so the model's Bridge is re-filed at once.

    The model's internal 17 lands on ground pads, decks and parts of the station buildings. It is
    re-filed straight after Rule A, before any other rule can read it:
      * at or below ``ground_max_hag`` -> Ground;
      * on a flat patch (``flat_cell`` XY cells with ``min_cell_points`` and a height spread of at
        most ``max_flat_std``): up to ``building_validation.min_roof_hag`` it is a pad -> Ground,
        above it a deck or roof -> Building (Rule L then validates it like any other Building);
      * anything else -> Unclassified, for the canopy and roof rules to decide.
    """
    b = spec["bridge_disposal"]
    bridge_code = 17
    idx = np.flatnonzero(classes == bridge_code)
    stats = {"enabled": bool(b["enabled"]), "bridge_points": int(len(idx))}
    if not bool(b["enabled"]) or not len(idx):
        print(f"        [Rule N] Bridge disposal: {len(idx):,} internal {bridge_code} pts"
              + ("" if b["enabled"] else " (disabled by contract)"))
        return stats
    hag = z_rel[idx]
    cell = np.floor(np.column_stack((x[idx], y[idx])) / float(b["flat_cell"])).astype(np.int64)
    inverse = np.asarray(np.unique(cell, axis=0, return_inverse=True)[1]).ravel()
    cnt = np.bincount(inverse)
    s1 = np.bincount(inverse, weights=z_rel[idx])
    s2 = np.bincount(inverse, weights=z_rel[idx] ** 2)
    std = np.sqrt(np.maximum(s2 / cnt - (s1 / cnt) ** 2, 0.0))
    flat = ((cnt >= int(b["min_cell_points"])) & (std <= float(b["max_flat_std"])))[inverse]
    roof_hag = float(spec["building_validation"]["min_roof_hag"])
    target = np.full(len(idx), 1, dtype=classes.dtype)
    target[flat & (hag > roof_hag)] = 6
    target[flat & (hag <= roof_hag)] = 2
    target[hag <= float(b["ground_max_hag"])] = 2
    classes[idx] = target
    provenance[idx] = ORIGIN_BRIDGE_DISPOSAL
    stats["to_ground"] = int(np.count_nonzero(target == 2))
    stats["to_building"] = int(np.count_nonzero(target == 6))
    stats["to_unclassified"] = int(np.count_nonzero(target == 1))
    print(f"        [Rule N] Bridge disposal: {len(idx):,} internal {bridge_code} pts -> Ground "
          f"{stats['to_ground']:,} (<= {float(b['ground_max_hag']):g} m or flat pad <= {roof_hag:g} m), "
          f"Building {stats['to_building']:,} (flat, above {roof_hag:g} m), Unclassified "
          f"{stats['to_unclassified']:,}")
    return stats


def rule_r_roof_consolidation(classes, z_rel, x, y, z_raw, spec, provenance):
    """Rule R: find each building by its roof and make the whole roof Building.

    The station buildings came out of the earlier rules in pieces - roof edges red (Rule F read
    them as cable), the rest grey or Bridge, and the canopy rules then turned what was left into
    vegetation. Here roofs are found from geometry alone, then consolidated:

    1. Roof points: above ``min_hag``, a ``k_neighbors`` neighbourhood that is a plane (planarity
       >= ``min_planarity``, sphericity <= ``max_sphericity``) facing up (|Nz| > ``min_normal_z``).
    2. Clusters of roof points (``cluster_cell`` grid, linked within ``cluster_link``) of at least
       ``min_area`` m2 are roofs, unless
         * at least ``max_ground_share`` of them are Ground - a rock ledge on a slope, not a roof
           (FUNIVIA000007: a 24 m2 ledge, 100 % Ground);
         * at least ``max_occupancy`` of the roof's cells have points under them, between 0.5 m
           HAG and 0.7 m below the roof. A roof hides what is under it, a hedge or a crown does
           not: on DU3039046 hedge rows measure 49-70 %, the station roofs 22-36 %.
    3. Every point inside the roof's 2D convex hull and within ``z_margin`` of the roof's height
       range becomes Building, and so do the walls under it above
       ``building_validation.min_roof_hag``; wire below that floor becomes Unclassified - all
       except poles, noise and confirmed wire.
    Returns ``(stats, footprint)``: ``footprint`` marks every point in the hull's column, so the
    canopy rules can be kept off the walls underneath.
    """
    r = spec["roof_consolidation"]
    footprint = np.zeros(len(classes), dtype=bool)
    stats = {"enabled": bool(r["enabled"]), "roofs": 0, "rejected_small": 0,
             "rejected_ground": 0, "rejected_occupied": 0, "consolidated": 0}
    if not bool(r["enabled"]):
        print("        [Rule R] Disabled by contract (roof_consolidation.enabled=false)")
        return stats, footprint
    if cKDTree is None or _ConvexHull is None:  # pragma: no cover - depends on the environment
        print("        [Rule R] [!] scipy is unavailable; roof consolidation skipped")
        return stats, footprint
    pole_code = int(spec["network_extraction"]["pole_class"])
    conductor_code = int(spec["network_extraction"]["conductor_class"])
    lap = _Laps()
    coords = np.column_stack((x, y, z_raw))
    finite = np.isfinite(coords).all(axis=1)
    cloud_idx = np.flatnonzero(finite & (z_rel > 0.15))
    cloud = coords[cloud_idx]
    tree = _tree(cloud)
    pos_of = np.full(len(classes), -1, dtype=np.int64)
    pos_of[cloud_idx] = np.arange(len(cloud_idx))
    frozen = np.isin(classes, (pole_code,) + NOISE_CODES)
    wire_idx, _ = _valid_conductor_lines(
        coords, np.flatnonzero(finite & (classes == conductor_code)), spec["canopy_homogenization"])
    frozen[wire_idx] = True
    lap("index")

    cand = np.flatnonzero(finite & (z_rel > float(r["min_hag"])) & ~frozen)
    if not len(cand):
        return stats, footprint
    k = int(min(int(r["k_neighbors"]), len(cloud)))
    normal_z = np.zeros(len(cand))
    plan = np.zeros(len(cand))
    sph = np.ones(len(cand))
    for start in range(0, len(cand), 200_000):
        stop = min(start + 200_000, len(cand))
        _, nb = tree.query(coords[cand[start:stop]], k=k, workers=-1)
        ev, vec = _neighbour_eig(cloud, np.asarray(nb).reshape(stop - start, k), k, vectors=True)
        l1 = np.maximum(ev[:, 2], 1e-12)
        plan[start:stop] = (ev[:, 1] - ev[:, 0]) / l1
        sph[start:stop] = np.maximum(ev[:, 0], 0.0) / l1
        normal_z[start:stop] = np.abs(vec[:, 2, 0])
    lap("shape")
    roof_pts = cand[(normal_z > float(r["min_normal_z"])) & (plan >= float(r["min_planarity"]))
                    & (sph <= float(r["max_sphericity"]))]
    if not len(roof_pts):
        print("        [Rule R] No planar, up-facing patches above "
              f"{float(r['min_hag']):g} m")
        return stats, footprint
    keys = np.floor(coords[roof_pts] / float(r["cluster_cell"])).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    nodes = coords[roof_pts][first]
    links = cKDTree(nodes).query_pairs(r=float(r["cluster_link"]), output_type="ndarray")
    group = _link_labels(len(nodes), links)[inverse]
    cell = float(r["cluster_cell"])
    flat_tree = _tree(coords[cloud_idx, :2])
    lap("cluster")
    g_order, g_start, g_count = _group_members(group)
    for g in np.flatnonzero(g_count):
        m = roof_pts[g_order[g_start[g]:g_start[g] + g_count[g]]]
        cells, ci = np.unique(np.floor(coords[m, :2] / cell).astype(np.int64), axis=0,
                              return_inverse=True)
        ci = np.asarray(ci).ravel()
        if len(cells) * cell * cell < float(r["min_area"]):
            stats["rejected_small"] += 1
            continue
        if np.mean(classes[m] == 2) >= float(r["max_ground_share"]):
            stats["rejected_ground"] += 1
            continue
        # under-roof occupancy: lowest roof point per cell vs. anything below it in that cell
        roof_z = np.full(len(cells), np.inf)
        np.minimum.at(roof_z, ci, coords[m, 2])
        lo, hi = coords[m, :2].min(axis=0), coords[m, :2].max(axis=0)
        box = cloud_idx[flat_tree.query_ball_point((lo + hi) / 2.0,
                                                   r=float(np.hypot(*(hi - lo))) / 2.0 + cell)]
        box = box[z_rel[box] > 0.5]
        bc = np.floor(coords[box, :2] / cell).astype(np.int64)
        lut = {tuple(c_): i for i, c_ in enumerate(cells)}
        cell_of = np.fromiter((lut.get(tuple(c_), -1) for c_ in bc), dtype=np.int64, count=len(bc))
        inside = cell_of >= 0
        below = inside.copy()
        below[inside] = coords[box[inside], 2] < roof_z[cell_of[inside]] - 0.7
        occupied = np.zeros(len(cells), dtype=bool)
        occupied[cell_of[below]] = True
        if occupied.mean() >= float(r["max_occupancy"]):
            stats["rejected_occupied"] += 1
            continue
        stats["roofs"] += 1
        hull_xy = coords[m, :2]
        try:
            hull = _ConvexHull(hull_xy)
        except Exception:  # degenerate (collinear) footprint
            continue
        poly = hull_xy[hull.vertices]
        col = box[_points_in_polygon(coords[box, :2], poly)]
        z_lo = float(coords[m, 2].min()) - float(r["z_margin"])
        z_hi = float(coords[m, 2].max()) + float(r["z_margin"])
        # The roof, plus the walls under it down to the Building floor (min_roof_hag): Rule F read
        # the upper station's facade as cable, and freezing the column alone kept it red. Below
        # that floor nothing is Building (the contract's pink purge), but no wire is either.
        roof_floor = float(spec["building_validation"]["min_roof_hag"])
        live = col[~frozen[col]]
        take = live[((coords[live, 2] >= z_lo) & (coords[live, 2] <= z_hi))
                    | ((coords[live, 2] < z_lo) & (z_rel[live] > roof_floor))]
        changed = take[classes[take] != 6]
        classes[take] = 6
        footprint[take] = True      # only what was taken: the hull also spans neighbouring trees
        provenance[changed] = ORIGIN_ROOF_CONSOLIDATION
        stats["consolidated"] += int(len(changed))
        low_wire = live[(z_rel[live] <= roof_floor) & (classes[live] == conductor_code)]
        classes[low_wire] = 1
        provenance[low_wire] = ORIGIN_ROOF_CONSOLIDATION
        stats["wall_wire_cleared"] = stats.get("wall_wire_cleared", 0) + int(len(low_wire))

    lap("roofs")
    # --- eaves and gutters: every Building instance, buffered outwards ---
    # A gutter is a straight edge (Rule F reads it as cable) and an eave is a step in the roof
    # (the crown tests read it as foliage), so the rim of a roof leaves the earlier rules red and
    # green. Each Building instance - every Building point now, from Rules B, L, N and the roofs
    # above, linked within ``eave_cluster_link`` - is given its 2D convex hull buffered by
    # ``eave_buffer`` (a true buffer: the hull of the hull's vertices offset in 16 directions),
    # and every point inside it, within ``eave_z_margin`` of the instance's height range and within
    # ``eave_buffer`` of one of its own points, becomes Building. Two limits stay: nothing at or
    # below the Building floor (min_roof_hag) - the contract's pink purge - and confirmed wire
    # whose line leaves the buffered outline (more than 1 - ``gutter_inside_share`` of it outside),
    # so the cable that reaches a station is not painted onto its roof; poles and noise are never
    # taken.
    eave = float(r["eave_buffer"])
    stats["eave_buildings"] = 0
    stats["eave_points"] = 0
    roof_floor = float(spec["building_validation"]["min_roof_hag"])
    if eave > 0:
        pole_noise = np.isin(classes, (pole_code,) + NOISE_CODES)
        confirmed = np.zeros(len(classes), dtype=bool)
        confirmed[wire_idx] = True
        # Which confirmed line each confirmed point belongs to (grid-decimated, as the detector
        # does), so a line can be judged whole: a gutter ends inside the buffered outline, the
        # cable to a station runs out of it.
        wire_label = np.full(len(classes), -1, dtype=np.int64)
        w_order = w_start = w_count = None
        if len(wire_idx):
            wkeys = np.floor(coords[wire_idx] / float(spec["canopy_homogenization"]["line_grid_cell"])
                             ).astype(np.int64)
            _, wfirst, winv = np.unique(wkeys, axis=0, return_index=True, return_inverse=True)
            wlinks = cKDTree(coords[wire_idx][wfirst]).query_pairs(
                r=float(spec["canopy_homogenization"]["line_link_radius"]), output_type="ndarray")
            wire_label[wire_idx] = _link_labels(len(wfirst), wlinks)[np.asarray(winv).ravel()]
            w_order, w_start, w_count = _group_members(wire_label[wire_idx])
        bidx = np.flatnonzero(finite & (classes == 6))
        if len(bidx) >= int(r["eave_min_points"]):
            keys = np.floor(coords[bidx] / float(r["cluster_cell"])).astype(np.int64)
            _, first_b, inv_b = np.unique(keys, axis=0, return_index=True, return_inverse=True)
            inv_b = np.asarray(inv_b).ravel()
            nodes_b = coords[bidx][first_b]
            links_b = cKDTree(nodes_b).query_pairs(r=float(r["eave_cluster_link"]),
                                                   output_type="ndarray")
            inst = _link_labels(len(nodes_b), links_b)[inv_b]
            # the tile's own density: median points per occupied 1 m2 cell, all classes
            cell_index, n_cells = _grid_cells(coords[finite, 0], coords[finite, 1], 1.0)
            cell_counts = np.bincount(cell_index, minlength=n_cells)
            tile_density = float(np.median(cell_counts[cell_counts > 0]))
            stats["tile_density"] = tile_density
            ang = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
            ring = np.column_stack((np.cos(ang), np.sin(ang))) * eave
            i_order, i_start, i_count = _group_members(inst)
            for g in np.flatnonzero(i_count):
                m = bidx[i_order[i_start[g]:i_start[g] + i_count[g]]]
                if len(m) < int(r["eave_min_points"]):
                    continue
                xy = coords[m, :2]
                try:
                    hull_core = _ConvexHull(xy)
                    core = xy[hull_core.vertices]
                    grown = (core[:, None, :] + ring[None, :, :]).reshape(-1, 2)
                    buffered = grown[_ConvexHull(grown).vertices]
                except Exception:  # degenerate (collinear) instance
                    continue
                # A roof is a surface and returns the tile's own point density; Building points
                # scattered through a tree row are a thin subset of foliage. They still link into
                # one "instance" through their 2 m neighbours, and the buffer would then paint the
                # forest around every one of them. CEDRASCO: three such instances (11 k, 6.6 k and
                # 5.5 k pts) held 3-12 pts per occupied m2 and took 280 k points, where the real
                # roofs there hold 100-330 against a tile median of 167 - and the hull fill is
                # no help, the scatter fills 72-89 % of its hull.
                occupied = len(np.unique(np.floor(xy).astype(np.int64), axis=0))
                if (len(m) / max(occupied, 1) < float(r["eave_min_density"]) * tile_density
                        or occupied < float(r["eave_min_fill"]) * max(hull_core.volume, 1.0)):
                    stats["eave_sparse"] = stats.get("eave_sparse", 0) + 1
                    continue
                lo, hi = buffered.min(axis=0), buffered.max(axis=0)
                box = cloud_idx[flat_tree.query_ball_point(
                    (lo + hi) / 2.0, r=float(np.hypot(*(hi - lo))) / 2.0 + 0.1)]
                inbuf = box[_points_in_polygon(coords[box, :2], buffered)]
                z_lo = float(coords[m, 2].min()) - float(r["eave_z_margin"])
                z_hi = float(coords[m, 2].max()) + float(r["eave_z_margin"])
                sel = inbuf[(coords[inbuf, 2] >= z_lo) & (coords[inbuf, 2] <= z_hi)
                            & (z_rel[inbuf] > roof_floor) & ~pole_noise[inbuf]]
                # Within ``eave_buffer`` of the building itself, not merely inside its hull: the
                # hull of a roof beside a tree row spans the trees, and filling the hull turned
                # whole crowns pink on CEDRASCO (286 k -> 841 k Building pts). An eave is a rim.
                if len(sel):
                    reach = cKDTree(coords[m]).query(coords[sel], k=1, distance_upper_bound=eave,
                                                     workers=-1)[0]
                    sel = sel[np.isfinite(reach)]
                # Confirmed wire stays unless its whole line sits inside the buffered outline:
                # a gutter is a straight edge of the building, and passes the line test like any
                # other straight edge. A line that runs out of the outline is a cable.
                cw = sel[confirmed[sel]]
                if len(cw):
                    gutter = []
                    for lab_ in np.unique(wire_label[cw]):
                        whole = wire_idx[w_order[w_start[lab_]:w_start[lab_] + w_count[lab_]]]
                        in_out = _points_in_polygon(coords[whole, :2], buffered)
                        if in_out.mean() >= float(r["gutter_inside_share"]):
                            gutter.append(lab_)
                    inside_core = _points_in_polygon(coords[sel, :2], core)
                    keep_wire = confirmed[sel] & ~np.isin(wire_label[sel], gutter) & ~inside_core
                    sel = sel[~keep_wire]
                changed = sel[classes[sel] != 6]
                classes[changed] = 6
                footprint[sel] = True
                provenance[changed] = ORIGIN_ROOF_CONSOLIDATION
                stats["eave_buildings"] += 1
                stats["eave_points"] += int(len(changed))
                stats.setdefault("eave_detail", []).append(
                    (int(len(m)), round(float(hull_core.volume), 1), round(occupied / max(hull_core.volume, 1.0), 2),
                     int(len(changed))))
    lap("eaves")
    print(f"        [Rule R] time by stage: {lap.line()}")
    print(f"        [Rule R] Eaves and gutters: {stats['eave_buildings']:,} Building instances "
          f"buffered by {eave:g} m (height +/- {float(r['eave_z_margin']):g} m, above "
          f"{roof_floor:g} m HAG; {stats.get('eave_sparse', 0):,} sparse scatters skipped, "
          f"< {float(r['eave_min_density']):g} x the tile's {stats.get('tile_density', 0):.0f} "
          f"pts/m2 or < {float(r['eave_min_fill']):.0%} of their hull) -> "
          f"{stats['eave_points']:,} rim pts -> Building")
    print(f"        [Rule R] Roof consolidation (planar, |Nz| > {float(r['min_normal_z']):g}, above "
          f"{float(r['min_hag']):g} m, >= {float(r['min_area']):g} m2): {stats['roofs']:,} roofs; "
          f"rejected {stats['rejected_small']:,} small, {stats['rejected_ground']:,} terrain ledges, "
          f"{stats['rejected_occupied']:,} with vegetation under them (occupancy >= "
          f"{float(r['max_occupancy']):.0%}); {stats['consolidated']:,} pts -> Building, "
          f"{int(footprint.sum()):,} pts in their footprints kept out of the canopy rules")
    return stats, footprint


def _points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd ray test, vectorised over the points (the polygon is a small convex hull)."""
    inside = np.zeros(len(pts), dtype=bool)
    px, py = pts[:, 0], pts[:, 1]
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        crosses = (y1 > py) != (y2 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (px < xint)
    return inside


def _trimmed_lsq(design: np.ndarray, values: np.ndarray, floor: float, iters: int = 4,
                 start: np.ndarray = None):
    """Least squares that re-fits on its own inliers only, so a few vegetation points in the
    corridor cannot drag the line off the wire. ``start`` optionally narrows the first fit.
    Returns ``(coefficients, inlier_mask)``."""
    keep = np.ones(len(values), dtype=bool) if start is None else np.asarray(start, dtype=bool)
    coef = np.zeros(design.shape[1])
    for _ in range(iters):
        if keep.sum() < design.shape[1] + 2:
            break
        coef, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
        res = values - design @ coef
        med = float(np.median(res[keep]))
        mad = float(np.median(np.abs(res[keep] - med)))
        limit = max(3.0 * 1.4826 * mad, floor)
        nxt = np.abs(res - med) <= limit
        if nxt.sum() < design.shape[1] + 2 or np.array_equal(nxt, keep):
            keep = nxt if nxt.sum() >= design.shape[1] + 2 else keep
            break
        keep = nxt
    return coef, keep


def _trace_span_fit(coords, tree_xy, a, b, t, ground_mask, classes=None, hag=None):
    """Fit the wire that runs between two poles to the raw points around their chord.

    The evidence is the cloud itself (any class but ground): wire labels cannot be the evidence,
    because in a forest they are exactly what is missing - and Rule F's own corridor stage fills
    the chord between two false towers with "wire", which would make the test circular.

    The curve is deliberately *not* hung from the two pole tops. A pole column stops wherever
    Rule F stopped, which on an angle pole can be the foot only, and the wire then hangs several
    metres above the recorded top; hung from the tops, the fit looks for the wire where the wire
    is not and finds nothing. So the wire is fitted where it actually is: a line of free height
    and free lateral drift, plus a sag, fitted to the corridor by a trimmed least square and then
    clamped to what a power line can do (slope, sag). What keeps that honest is not where the fit
    landed but the canopy test - a bin counts only when the thin layer stands out from the
    vegetation above and below it. Returns ``(coverage, offset, sag, curve_points, z_a, z_b)``.
    """
    empty = np.zeros(0, dtype=np.int64)
    run = float(np.hypot(*(b[:2] - a[:2])))
    if run < 1e-6:
        return 0.0, 0.0, 0.0, empty, float(a[2]), float(b[2])
    dz_tube, tube = float(t["tube_dz"]), float(t["tube"])
    gap, wid = float(t["ambient_gap"]), float(t["ambient_width"])
    stand = float(t["stand_out"])
    n_bin = max(int(run // float(t["bin"])), 3)
    fr_all = (np.arange(n_bin) + 0.5) / n_bin
    # The ends of a span are the pole hardware - cross-arms, insulators, the column itself - not a
    # clean catenary, so coverage is only scored clear of the poles by ``end_margin``.
    clear = (fr_all * run >= float(t["end_margin"])) & ((1.0 - fr_all) * run >= float(t["end_margin"]))
    if not clear.any():
        clear = np.zeros(n_bin, dtype=bool)
        clear[1:-1] = True
    fr = fr_all[clear]
    axis = (b[:2] - a[:2]) / run
    left = np.array([-axis[1], axis[0]])
    samples = a[:2] + (b[:2] - a[:2]) * fr[:, None]
    wide, half_bin = float(t["corridor_wide"]), float(t["bin"]) / 2.0
    # Only the air above the ground is searched. A span is a wire in the air; without this a flat
    # surface a metre off the ground - a water course, a roof, a road - is a thin layer with
    # nothing above or below it, and the canopy test below cannot tell it from a wire.
    if hag is not None:
        low_air = hag < float(t["min_span_hag"])
    else:
        low_air = np.zeros(len(coords), dtype=bool)
    ids, bins = [], []
    for bi, h in enumerate(tree_xy.query_ball_point(samples, r=wide + half_bin)):
        if not len(h):
            continue
        h = np.asarray(h, dtype=np.int64)
        h = h[~ground_mask[h] & ~low_air[h]]
        if not len(h):
            continue
        rel = coords[h, :2] - a[:2]
        sel = (np.abs(rel @ left) <= wide) & (np.abs(rel @ axis - fr[bi] * run) <= half_bin)
        if sel.any():
            ids.append(h[sel])
            bins.append(np.full(int(sel.sum()), bi, dtype=np.int64))
    if not ids:
        return 0.0, 0.0, 0.0, empty, float(a[2]), float(b[2])
    pid = np.concatenate(ids)
    pbin = np.concatenate(bins)
    pz = coords[pid, 2]
    plat = (coords[pid, :2] - a[:2]) @ left
    along = fr[pbin] * run
    sag_b = 4.0 * fr[pbin] * (1.0 - fr[pbin])
    ones = np.ones(len(pid))
    az = np.column_stack((ones, along, sag_b))
    slope_max, sag_max = float(t["max_slope"]), float(t["max_sag_ratio"]) * run

    # Seed the fit from the layers that already stand out in the corridor. A span hangs over
    # ground and vegetation, and the bulk of its corridor is that clutter, so a fit started from
    # the median of the corridor slides down onto it instead of onto the wire.
    cand = np.arange(float(pz.min()), float(pz.max()) + dz_tube, dz_tube)
    dev = np.abs(pz[None, :] - cand[:, None])
    score = (dev <= dz_tube).sum(axis=1) - stand * 0.5 * (
        ((dev > gap) & (dev <= gap + wid)).sum(axis=1)
        + ((dev < -gap) & (dev >= -gap - wid)).sum(axis=1))
    seeds = []
    for ci in np.argsort(score)[::-1]:
        if score[ci] <= 0.0 or len(seeds) >= int(t["fit_seeds"]):
            break
        if all(abs(cand[ci] - s) > 1.0 for s in seeds):
            seeds.append(float(cand[ci]))
    if not seeds:
        return 0.0, 0.0, 0.0, empty, float(a[2]), float(b[2])

    best = None
    for z0 in seeds:
        # z = c0 + c1 * along - sag * sag_basis, trimmed onto its own inliers each pass, then
        # clamped to what a power line can do: a limited slope, and a sag that pulls down.
        coef = np.array([z0, 0.0, 0.0])
        keep = np.abs(pz - z0) <= max(4.0 * dz_tube, 0.8)
        fitted = True
        for _ in range(6):
            res = pz - az @ coef
            med = float(np.median(res[keep])) if keep.any() else 0.0
            mad = float(np.median(np.abs(res[keep] - med))) if keep.any() else 0.0
            keep = np.abs(res - med) <= max(3.0 * 1.4826 * mad, dz_tube)
            if keep.sum() < 6:
                fitted = False
                break
            coef, *_ = np.linalg.lstsq(az[keep], pz[keep], rcond=None)
            coef[1] = float(np.clip(coef[1], -slope_max, slope_max))
            coef[2] = float(np.clip(coef[2], -sag_max, 0.0))
        if not fitted:
            continue
        res = pz - az @ coef
        # The wire does not run down the middle of the pole chord: it drifts sideways from one
        # cross-arm to the other, by several metres over a span. So the layer is given its own
        # lateral line, fitted to the points the vertical fit just claimed.
        layer = np.abs(res) <= dz_tube
        if layer.sum() >= 4:
            lcoef, _ = _trimmed_lsq(np.column_stack((ones, along)), plat, tube, start=layer)
        else:
            lcoef = np.array([0.0, 0.0])
        lat_ok = np.abs(plat - (lcoef[0] + lcoef[1] * along)) <= tube
        band = lat_ok & layer
        per_bin = np.bincount(pbin[band], minlength=len(fr))
        # A wire is a thin layer in clear air; a crown is dense above and below the same curve.
        # Two bands of the same width, just outside the wire layer on either side, give the
        # ambient level, and a bin only counts when the layer stands out from it. The bands must
        # sit outside the layer: overlap them with it and they measure the wire itself, which
        # makes the test vacuous exactly where it is needed.
        inner = max(gap, dz_tube)
        up = lat_ok & (res > inner) & (res <= inner + wid)
        down = lat_ok & (res < -inner) & (res >= -inner - wid)
        ambient = 0.5 * (np.bincount(pbin[up], minlength=len(fr))
                         + np.bincount(pbin[down], minlength=len(fr)))
        good = per_bin >= np.maximum(int(t["bin_points"]), stand * ambient)
        cov = float(np.count_nonzero(good)) / len(fr)
        if best is None or cov > best[0]:
            best = (cov, coef.copy(), lcoef.copy())
    if best is None:
        return 0.0, 0.0, 0.0, empty, float(a[2]), float(b[2])
    cov, coef, lcoef = best
    sag = float(-coef[2])
    curve = coef[0] + coef[1] * along - sag * sag_b
    lat_fit = lcoef[0] + lcoef[1] * along
    res = pz - curve
    lat_off = plat - lat_fit
    keep = (np.abs(res) <= dz_tube) & (np.abs(lat_off) <= tube)
    if classes is not None:
        # Points the model filed as noise around a wire are its other phases and strands: they hang
        # a little lower and a little to the side of the fitted line, so the band is wider for them.
        # They are still claimed only inside the wider band, never from the canopy the corridor
        # crosses.
        noise = np.isin(classes[pid], tuple(int(c_) for c_ in t["noise_codes"]))
        keep = keep | (noise & (np.abs(lat_off) <= float(t["tube_wide"]))
                       & (res <= dz_tube) & (res >= -float(t["tube_wide_below"])))
    z_a, z_b = float(coef[0]), float(coef[0] + coef[1] * run)
    return cov, float(coef[0] - a[2]), sag, pid[keep], z_a, z_b


def rule_t_network_trace(classes, z_rel, x, y, z_raw, spec, provenance):
    """Rule T: follow the line from poles that are certainly poles, so the canopy cannot hide it.

    Rules I and J judge a pole or a wire by what surrounds it, so a pole or a span that runs
    through trees looks like canopy and is lost, while the same line in the open is kept. A line
    is a connected thing, though: if one pole is certain and the wire leaves it towards the next
    pole, that next pole is certain too, whatever grows around it. This rule walks the network
    instead of judging every part alone:

    1. Towers: Rule F's pole points linked in 2D into columns, kept when tall, thin and with
       enough points.
    2. Seeds: towers that stand in the open - nothing but the pole within ``shaft_outer`` of its
       own shaft, between 1.5 m above its foot and its cross-arm. In a forest a crown cylinder is
       never like that, a pole always is, and a bush two metres off a pole does not stop it being
       one.
    3. Walk. From a seed every tower at ``min_span``..``max_span`` is tried, because on an angle
       pole the wire fragment there is the blend of the two spans meeting and its axis points at
       neither neighbour; which one is real is decided by the fit, not by the heading. Once a
       tower is reached the walk is straight on (``max_bend``), and each span is accepted when the
       raw points show a wire running between the two (see ``_trace_span_fit``). The walk
       continues from every tower it reaches, until no tower is found ahead.
    4. Claims. A traced tower keeps its pole points and also takes ``claim_classes`` points inside
       its ``flood_radius`` cylinder, up to the height the wire meets it. A traced span keeps its
       wire points and takes the ``claim_classes`` points inside its thin tube.

    The poles and wire found are returned so Rules I and J leave them alone. Nothing outside a
    traced path is touched.
    """
    t = spec["network_trace"]
    empty = {"enabled": False, "poles": 0, "spans": 0, "seeds": 0,
             "pole_idx": np.zeros(0, dtype=np.int64), "wire_idx": np.zeros(0, dtype=np.int64)}
    if not bool(t["enabled"]):
        print("        [Rule T] Disabled by contract (network_trace.enabled=false)")
        return empty
    if cKDTree is None:  # pragma: no cover - depends on the environment
        print(f"        [Rule T] [!] scipy is unavailable ({_SCIPY_ERROR}); network trace skipped")
        return empty
    pole_code = int(spec["network_extraction"]["pole_class"])
    cond_code = int(spec["network_extraction"]["conductor_class"])
    stats = dict(empty, enabled=True, columns=0, towers=0, open_towers=0, traced_spans=0,
                 reached_beyond_seeds=0, claimed_pole_points=0, claimed_wire_points=0)

    coords = np.column_stack((x, y, z_raw))
    finite = np.isfinite(coords).all(axis=1)
    pole_idx = np.flatnonzero(finite & (classes == pole_code))
    wire_idx = np.flatnonzero(finite & (classes == cond_code))
    if len(pole_idx) < 2:
        print(f"        [Rule T] {len(pole_idx):,} pole pts: nothing to trace")
        return stats

    # --- 1. towers ---
    cell = float(t["column_cell"])
    cell_key = np.floor(coords[pole_idx, :2] / cell).astype(np.int64)
    _, first, inverse = np.unique(cell_key, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    nodes = coords[pole_idx[first], :2]
    links = cKDTree(nodes).query_pairs(r=float(t["column_link"]), output_type="ndarray")
    column = _link_labels(len(nodes), links)[inverse]
    n_col = int(column.max()) + 1
    order = np.argsort(column, kind="stable")
    counts = np.bincount(column, minlength=n_col)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    pts = coords[pole_idx[order]]
    stats["columns"] = int(np.count_nonzero(counts))
    z_lo = np.minimum.reduceat(pts[:, 2], starts)
    z_hi = np.maximum.reduceat(pts[:, 2], starts)
    mean_xy = np.column_stack([np.add.reduceat(pts[:, d], starts) for d in range(2)]) / counts[:, None]
    width = np.maximum(np.maximum.reduceat(pts[:, 0], starts) - np.minimum.reduceat(pts[:, 0], starts),
                       np.maximum.reduceat(pts[:, 1], starts) - np.minimum.reduceat(pts[:, 1], starts))
    top_hag = np.maximum.reduceat(z_rel[pole_idx[order]], starts)
    good = ((counts >= int(t["min_points"])) & ((z_hi - z_lo) >= float(t["min_height"]))
            & (width <= float(t["max_width"])) & (top_hag >= float(t["min_hag"])))
    tower_col = np.flatnonzero(good)
    stats["towers"] = int(len(tower_col))
    if len(tower_col) < 2:
        print(f"        [Rule T] {stats['columns']:,} pole columns, {len(tower_col)} look like a "
              "pole: nothing to trace")
        return stats
    heads = np.zeros((len(tower_col), 3))
    for ti, c_ in enumerate(tower_col):
        seg = pts[starts[c_]:starts[c_] + counts[c_]]
        top = seg[seg[:, 2] >= z_hi[c_] - 2.0]
        heads[ti] = (top[:, 0].mean(), top[:, 1].mean(), z_hi[c_])
    tower_xy = mean_xy[tower_col]

    # --- 2. seeds: towers standing in the open ---
    finite_idx = np.flatnonzero(finite)
    flat_tree = _tree(coords[finite_idx, :2], leafsize=32)
    ring_lo, shaft_out = float(t["ring_inner"]), float(t["shaft_outer"])
    seed = np.zeros(len(tower_col), dtype=bool)
    for ti, c_ in enumerate(tower_col):
        centre = tower_xy[ti]
        near_pts = finite_idx[np.asarray(flat_tree.query_ball_point(centre, r=shaft_out), dtype=np.int64)]
        d = np.hypot(coords[near_pts, 0] - centre[0], coords[near_pts, 1] - centre[1])
        zz = coords[near_pts, 2]
        # Only the shaft itself is asked about, out to ``shaft_outer``: a pole with a bush standing
        # two metres off it is still a pole, and a crown cylinder is solid right up to its axis.
        between = ((d >= ring_lo) & (zz >= z_lo[c_] + float(t["ring_foot"])) & (zz <= z_hi[c_] - 1.0)
                   & ~np.isin(classes[near_pts], (2, pole_code, cond_code)))
        seed[ti] = int(between.sum()) <= float(t["open_ratio"]) * counts[c_]
    stats["open_towers"] = int(seed.sum())
    if not seed.any():
        print(f"        [Rule T] {len(tower_col)} towers, none stands in the open: nothing to trace from")
        return stats

    # --- 3. walk ---
    wire_tree = cKDTree(coords[wire_idx]) if len(wire_idx) else None
    ground_mask = classes == 2
    head_tree = cKDTree(heads[:, :2])
    trusted = seed.copy()
    spans = []                              # (i, j, offset, sag, curve points, z_i, z_j)
    queue = []                              # (tower, unit heading, max turn, known, heading is a gate)
    for ti in np.flatnonzero(seed):
        axis = None
        if wire_tree is not None:
            frag = wire_idx[np.asarray(wire_tree.query_ball_point(heads[ti], r=float(t["seed_fragment_radius"])),
                                       dtype=np.int64)]
            if len(frag) >= int(t["seed_fragment_points"]):
                xy = coords[frag, :2] - coords[frag, :2].mean(axis=0)
                vals, vecs = np.linalg.eigh(np.cov(xy.T))
                if vals[1] > 0 and vals[0] / vals[1] < float(t["seed_fragment_flatness"]):
                    axis = vecs[:, 1]
        # The fragment is a score, not a gate. On an angle pole it is the blend of the two spans
        # meeting there, so its axis points at neither neighbour, and gating on it would lose the
        # line exactly where it turns. Which neighbour is real is decided by the fit.
        queue.append((ti, axis, 180.0, axis is not None, False))
    seen_edges = set()
    mapped = flat_tree_all(flat_tree, finite_idx)
    while queue:
        ti, heading, turn, known, gated = queue.pop(0)
        a = heads[ti]
        cand = np.asarray(head_tree.query_ball_point(a[:2], r=float(t["max_span"])), dtype=np.int64)
        best = None
        for tj in cand:
            if tj == ti:
                continue
            b = heads[tj]
            vec = b[:2] - a[:2]
            run = float(np.hypot(*vec))
            if run < float(t["min_span"]) or run > float(t["max_span"]):
                continue
            cosang = float(vec @ heading) / run if heading is not None else 1.0
            if gated and cosang < np.cos(np.radians(turn)):
                continue
            key = (min(ti, int(tj)), max(ti, int(tj)))
            if key in seen_edges:
                continue
            cov, offset, sag, claim, za, zb = _trace_span_fit(coords, mapped, a, b, t,
                                                                ground_mask, classes, z_rel)
            need = float(t["min_coverage"]) if known else float(t["min_coverage_blind"])
            if cov >= need:
                score = cov - 0.002 * run * (1.0 - cosang)      # straighter and better covered
                if best is None or score > best[0]:
                    best = (score, int(tj), offset, sag, key, claim, za, zb)
        if best is None:
            continue
        _, tj, offset, sag, key, claim, za, zb = best
        seen_edges.add(key)
        spans.append((ti, tj, offset, sag, claim, za, zb))
        if not trusted[tj]:
            trusted[tj] = True
            vec = heads[tj, :2] - a[:2]
            queue.append((tj, vec / np.hypot(*vec), float(t["max_bend"]), True, True))
    stats["poles"] = int(trusted.sum())
    stats["traced_spans"] = int(len(spans))
    stats["reached_beyond_seeds"] = int(trusted.sum() - seed.sum())
    if not spans:
        print(f"        [Rule T] {len(tower_col)} towers, {int(seed.sum())} stand in the open, but no "
              "span fits a catenary from any of them")
        return stats

    # --- 4. claims ---
    claim_cls = tuple(int(c_) for c_ in t["claim_classes"])
    # What a hidden span may take: grey and the two noise classes - the wire's own other phases
    # and strands. Vegetation is NOT wire. A span may not eat the canopy it crosses, or the whole
    # forest the line runs through turns red; the wire's real points there are recovered by the
    # wide noise band in _trace_span_fit, not by claiming every green point near the curve.
    wire_claim_cls = tuple(int(c_) for c_ in t["wire_claim_classes"])
    # Where the wire meets each traced tower, so a pole whose column stopped low is still claimed
    # up to its real cross-arm instead of to the top of what Rule F happened to find.
    attach = np.full(len(tower_col), -np.inf)
    for ti, tj, _o, _s, _c, za, zb in spans:
        attach[ti] = max(attach[ti], za)
        attach[tj] = max(attach[tj], zb)
    keep_pole = []
    flood_r = float(t["flood_radius"])
    max_lift = float(t["max_attach_lift"])
    for ti in np.flatnonzero(trusted):
        c_ = tower_col[ti]
        keep_pole.append(pole_idx[order[starts[c_]:starts[c_] + counts[c_]]])
        z_top = z_hi[c_] + 0.5
        if np.isfinite(attach[ti]):
            z_top = max(z_top, min(attach[ti] + 0.5, z_hi[c_] + max_lift))
        cand = finite_idx[np.asarray(flat_tree.query_ball_point(tower_xy[ti], r=flood_r), dtype=np.int64)]
        cand = cand[np.isin(classes[cand], claim_cls) & (coords[cand, 2] >= z_lo[c_] - 0.2)
                    & (coords[cand, 2] <= z_top)]
        if len(cand):
            keep_pole.append(cand)
    pole_hit = np.unique(np.concatenate(keep_pole))
    claimed_pole = pole_hit[classes[pole_hit] != pole_code]
    classes[claimed_pole] = pole_code
    provenance[claimed_pole] = ORIGIN_POLE
    stats["claimed_pole_points"] = int(len(claimed_pole))

    wire_keep = [claim[np.isin(classes[claim], wire_claim_cls + (cond_code,))]
                 for _i, _j, _o, _s, claim, _za, _zb in spans if len(claim)]
    wire_hit = np.unique(np.concatenate(wire_keep)) if wire_keep else np.zeros(0, dtype=np.int64)
    claimed_wire = wire_hit[classes[wire_hit] != cond_code]
    classes[claimed_wire] = cond_code
    provenance[claimed_wire] = ORIGIN_CONDUCTOR
    stats["claimed_wire_points"] = int(len(claimed_wire))
    stats["pole_idx"] = pole_hit
    stats["wire_idx"] = wire_hit
    stats["_heads"], stats["_trusted"], stats["_seed"], stats["_spans"] = heads, trusted, seed, \
        [sp[:4] for sp in spans]
    print(f"        [Rule T] Network trace: {stats['columns']:,} pole columns, {stats['towers']:,} look "
          f"like a pole, {stats['open_towers']:,} stand in the open")
    print(f"                  walked from those along the wire: {stats['poles']:,} poles "
          f"({stats['reached_beyond_seeds']:,} reached) and {stats['traced_spans']:,} spans")
    print(f"                  protected: {len(pole_hit):,} pole pts ({stats['claimed_pole_points']:,} "
          f"claimed) and {len(wire_hit):,} wire pts ({stats['claimed_wire_points']:,} claimed)")
    return stats


def flat_tree_all(tree, finite_idx):
    """Adapter: a KD-tree over the finite points that answers in the full point numbering."""
    class _Mapped:
        def query_ball_point(self, pts, r):
            out = tree.query_ball_point(pts, r=r)
            return [finite_idx[np.asarray(o, dtype=np.int64)] for o in out]
    return _Mapped()


def purge_wire_blobs(classes, z_rel, x, y, z_raw, spec, provenance):
    """Wire that is a small blob inside a crown is foliage, not a cable.

    A conductor is a line metres long. What is left of the wire class after the canopy and pole
    rules also holds compact clumps a few metres across, in the middle of a crown (on CEDRASCO 16
    of them, 2-4 m wide, with 80-95 % vegetation around them): they showed as red rings in the
    canopy, and the Unclassified points inside them stayed grey because nothing may touch the
    points right next to a wire. Wire points are linked in 3D (``link``); a component whose
    horizontal extent stays under ``max_extent``, with at least ``min_veg_share`` vegetation among
    the points within ``veg_radius`` of its centre, and with no pole point within
    ``pole_keep_radius`` (a stub on a cross-arm is real), becomes vegetation of its height band.
    """
    w = spec["wire_blob_purge"]
    if not bool(w["enabled"]):
        print("        [Rule X] Disabled by contract (wire_blob_purge.enabled=false)")
        return {"enabled": False, "blobs": 0, "points": 0}
    if cKDTree is None:  # pragma: no cover - depends on the environment
        return {"enabled": False, "blobs": 0, "points": 0}
    cond = int(spec["network_extraction"]["conductor_class"])
    pole = int(spec["network_extraction"]["pole_class"])
    v = spec["vegetation_split"]
    lo_max, med_max = float(v["low_veg_max_hag"]), float(v["med_veg_max_hag"])
    coords = np.column_stack((x, y, z_raw))
    finite = np.isfinite(coords).all(axis=1)
    wire = np.flatnonzero(finite & (classes == cond))
    stats = {"enabled": True, "components": 0, "blobs": 0, "points": 0}
    if not len(wire):
        return stats
    links = cKDTree(coords[wire]).query_pairs(r=float(w["link"]), output_type="ndarray")
    label = _link_labels(len(wire), links)
    order = np.argsort(label, kind="stable")
    counts = np.bincount(label)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    pts = coords[wire[order]]
    ext = np.maximum(np.maximum.reduceat(pts[:, 0], starts) - np.minimum.reduceat(pts[:, 0], starts),
                     np.maximum.reduceat(pts[:, 1], starts) - np.minimum.reduceat(pts[:, 1], starts))
    stats["components"] = int(np.count_nonzero(counts))
    small = np.flatnonzero((counts > 0) & (ext < float(w["max_extent"])))
    if not len(small):
        print(f"        [Rule X] {stats['components']:,} wire components, none is a small blob")
        return stats
    centre = np.column_stack([np.add.reduceat(pts[:, d], starts) for d in range(3)]) / np.maximum(counts, 1)[:, None]
    all_tree = _tree(coords[finite])
    fin_idx = np.flatnonzero(finite)
    pole_pts = np.flatnonzero(finite & (classes == pole))
    pole_tree = cKDTree(coords[pole_pts]) if len(pole_pts) else None
    take = []
    for comp in small:
        members = wire[order[starts[comp]:starts[comp] + counts[comp]]]
        if pole_tree is not None:
            d = pole_tree.query(coords[members], k=1, distance_upper_bound=float(w["pole_keep_radius"]))[0]
            if np.any(np.asarray(d) < np.inf):
                continue
        near = fin_idx[np.asarray(all_tree.query_ball_point(centre[comp], r=float(w["veg_radius"])),
                                  dtype=np.int64)]
        near = near[classes[near] != cond]
        if not len(near):
            continue
        if float(np.isin(classes[near], VEG_CODES).mean()) >= float(w["min_veg_share"]):
            take.append(members)
    if take:
        picked = np.concatenate(take)
        h = z_rel[picked]
        classes[picked] = np.where(h <= lo_max, 3, np.where(h <= med_max, 4, 5)).astype(classes.dtype)
        provenance[picked] = ORIGIN_CANOPY
        stats["blobs"], stats["points"] = len(take), int(len(picked))
    print(f"        [Rule X] {stats['components']:,} wire components, {len(small):,} narrower than "
          f"{float(w['max_extent']):g} m; {stats['blobs']:,} of them sit in a crown "
          f"(>= {float(w['min_veg_share']):.0%} vegetation within {float(w['veg_radius']):g} m, no pole) "
          f"-> {stats['points']:,} pts to vegetation")
    return stats


def rule_v_low_vegetation_fill(classes, z_rel, x, y, z_raw, spec, provenance):
    """Rule V: give the low layer and the gaps in the canopy back to vegetation.

    1. Low layer. The model labels hardly anything below a metre or two as vegetation: grass,
       crops and shrubs come out of it as Unclassified (on CEDRASCO 111 k points between 0.15 and
       1.0 m, against 14 k it called vegetation), and the canopy rule (Rule I) only fills from
       ``fill_min_hag`` (2 m) up. A point still in ``source_classes`` after every other rule,
       standing between ``spurious_veg_min_hag`` and ``max_hag`` above the terrain, is filed into
       the vegetation band of its height (Rule A's edges).
    2. Canopy gaps. Higher up, Unclassified clusters stay inside crowns where Rule I could not
       reach them - typically under a wire, whose corridor Rule I must leave alone. Such a point
       becomes vegetation of its band when at least ``canopy_veg_share`` of the classified points
       within ``canopy_radius`` (3D) are vegetation; this is repeated ``canopy_passes`` times so a
       gap closes from its edge inwards.

    Kept out in both: anything within ``structure_buffer`` (3D) of a building, pole or wire
    point - a wall foot, the base of a pole or the wire itself. The distance is 3D on purpose: a
    shrub under an overhead line is still a shrub (a 2D buffer kept every bush under the lines
    Unclassified, which showed as grey patches inside the vegetation).
    """
    v = spec["low_vegetation_fill"]
    if not bool(v["enabled"]):
        print("        [Rule V] Disabled by contract (low_vegetation_fill.enabled=false)")
        return {"enabled": False, "filled": 0}
    split = spec["vegetation_split"]
    lo_min = float(split["spurious_veg_min_hag"])
    lo_max, med_max = float(split["low_veg_max_hag"]), float(split["med_veg_max_hag"])
    top = min(float(v["max_hag"]), med_max) if float(v["max_hag"]) > 0 else med_max
    sources = tuple(int(c) for c in v["source_classes"])
    finite = np.isfinite(z_rel) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z_raw)
    src = np.isin(classes, sources) & finite & (z_rel > lo_min)
    stats = {"enabled": True, "candidates": int(src.sum()), "near_structure": 0, "filled": 0,
             "low": 0, "medium": 0, "high": 0, "canopy_gap": 0}
    if not src.any():
        print("        [Rule V] No Unclassified points above the terrain")
        return stats

    coords = np.column_stack((x, y, z_raw))
    structure_codes = (6, int(spec["network_extraction"]["pole_class"]),
                       int(spec["network_extraction"]["conductor_class"]))
    struct = np.flatnonzero(np.isin(classes, structure_codes) & finite)
    buf = float(v["structure_buffer"])
    if len(struct) and buf > 0 and cKDTree is not None:
        idx = np.flatnonzero(src)
        d = cKDTree(coords[struct]).query(coords[idx], k=1, distance_upper_bound=buf, workers=-1)[0]
        near = idx[np.asarray(d) < np.inf]
        src[near] = False
        stats["near_structure"] = int(len(near))

    def band(sel):
        h = z_rel[sel]
        return np.where(h <= lo_max, 3, np.where(h <= med_max, 4, 5)).astype(classes.dtype)

    # Buildings get a 2D buffer on top of the 3D one: a wall under its eaves, the base course and
    # the steps along a facade are metres below the roof points but still part of the building.
    bld = np.flatnonzero(finite & (classes == 6))
    bbuf = float(v["building_buffer_2d"])
    stats["near_building"] = 0
    if len(bld) and bbuf > 0 and cKDTree is not None and src.any():
        idx = np.flatnonzero(src)
        d = cKDTree(coords[bld, :2]).query(coords[idx, :2], k=1, distance_upper_bound=bbuf,
                                           workers=-1)[0]
        near = idx[np.asarray(d) < np.inf]
        src[near] = False
        stats["near_building"] = int(len(near))

    # Man-made shapes are not shrubs: a kerb, a fence or a garden wall is a line or a vertical
    # plane of points, vegetation is a scatter. The shape of each point's neighbourhood (its k
    # nearest points above the terrain) decides; such points stay Unclassified.
    stats["man_made"] = 0
    k_geo = int(v["shape_k"])
    if src.any() and k_geo >= 4 and cKDTree is not None:
        above = np.flatnonzero(finite & (z_rel > lo_min) & (classes != 2))
        cand = np.flatnonzero(src)
        dist, nb = cKDTree(coords[above]).query(coords[cand], k=k_geo,
                                                distance_upper_bound=float(v["shape_radius"]),
                                                workers=-1)
        full = np.isfinite(dist).all(axis=1)
        if full.any():
            evals, evecs = _neighbour_eig(coords[above], nb[full], k_geo, vectors=True)
            l3, l2, l1 = evals[:, 0], evals[:, 1], evals[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                linear = (l1 - l2) / l1 >= float(v["max_linearity"])
                planar = (l2 - l3) / l1 >= float(v["max_planarity"])
            upright = np.abs(evecs[:, 2, 0]) <= float(v["wall_max_nz"])   # normal is horizontal
            # a smooth surface off the ground - a flat or pitched roof of a shed, a carport, a car -
            # is not foliage either (grass is a plane too, but it lies at the terrain)
            raised = z_rel[cand[full]] >= float(v["roof_min_hag"])
            man = linear | (planar & (upright | raised))
            hit = cand[full][man]
            src[hit] = False
            stats["man_made"] = int(len(hit))

    # --- 1. the low layer ---
    low_layer = src & (z_rel <= top)
    classes[low_layer] = band(low_layer)
    provenance[low_layer] = ORIGIN_VEG_SPLIT

    # --- 2. gaps in the canopy ---
    gap = src & (z_rel > top)
    filled = low_layer.copy()
    if gap.any() and cKDTree is not None and int(v["canopy_passes"]) > 0:
        gap_idx = np.flatnonzero(gap)
        tree = _tree(coords[finite])
        fin_idx = np.flatnonzero(finite)
        hits = tree.query_ball_point(coords[gap_idx], r=float(v["canopy_radius"]), workers=-1)
        lens = np.fromiter(map(len, hits), dtype=np.int64, count=len(hits))
        flat = fin_idx[np.fromiter(chain.from_iterable(hits), dtype=np.int64, count=int(lens.sum()))]
        owner = np.repeat(np.arange(len(gap_idx)), lens)
        share_min = float(v["canopy_veg_share"])
        min_veg = int(v["canopy_min_veg"])
        open_ = np.ones(len(gap_idx), dtype=bool)
        for _ in range(int(v["canopy_passes"])):
            nb_cls = classes[flat]
            is_veg = np.isin(nb_cls, VEG_CODES)
            classified = ~np.isin(nb_cls, sources)
            n_veg = np.bincount(owner, weights=is_veg, minlength=len(gap_idx))
            n_cls = np.bincount(owner, weights=classified, minlength=len(gap_idx))
            take = open_ & (n_veg >= min_veg) & (n_veg >= share_min * np.maximum(n_cls, 1))
            if not take.any():
                break
            pick = gap_idx[take]
            classes[pick] = band(pick)
            provenance[pick] = ORIGIN_CANOPY
            filled[pick] = True
            open_ &= ~take
            stats["canopy_gap"] += int(take.sum())

    stats["low"] = int(np.count_nonzero(filled & (classes == 3)))
    stats["medium"] = int(np.count_nonzero(filled & (classes == 4)))
    stats["high"] = int(np.count_nonzero(filled & (classes == 5)))
    stats["filled"] = int(filled.sum())
    print(f"        [Rule V] Unclassified above {lo_min:g} m: {stats['candidates']:,} pts; "
          f"{stats['near_structure']:,} within {buf:g} m (3D) of a building, pole or wire and "
          f"{stats['near_building']:,} within {bbuf:g} m (2D) of a building kept as they are; "
          f"{stats['man_made']:,} in a line or a wall (kerb, fence) kept as they are")
    print(f"                  low layer (<= {top:g} m) and {stats['canopy_gap']:,} canopy-gap pts -> "
          f"{stats['low']:,} low, {stats['medium']:,} medium, {stats['high']:,} high vegetation")
    return stats


def rule_g_ground_refinement(classes, z_rel, x, y, spec, provenance, after_thinning=False):
    """Rule G: fold terrain-hugging speckle back into Ground.

    Water and unclassified returns that hug the terrain surface (HAG below
    ``ground_refinement.max_hag``) are the stray coloured points a cross-section shows above
    and on the ground. The client's 10-class schema wants them as Ground, so they are absorbed.
    A large, contiguous sheet of them is a water body rather than speckle, so the density of the
    absorbed points is measured and reported instead of being assumed away.

    Points Rule D deliberately unclassified are never re-absorbed: otherwise,
    ``scope=after_thinning`` would undo the thinning sieve point for point.
    """
    g = spec["ground_refinement"]
    scope = "after_thinning" if after_thinning else "before_thinning"
    if not bool(g["enabled"]):
        print(f"        [Rule G] Disabled by contract (ground_refinement.enabled=false, "
              f"scope={g['scope']})")
        return {"enabled": False, "absorbed": 0, "scope": scope}
    sources = tuple(int(c) for c in g["source_classes"])
    target = int(g["target_class"])
    max_hag = float(g["max_hag"])

    source_mask = np.isin(classes, sources)
    near_terrain = z_rel <= max_hag
    protected = provenance == ORIGIN_GROUND_THINNING
    mask = source_mask & near_terrain & ~protected
    per_source = {int(c): int(np.count_nonzero(mask & (classes == c))) for c in sources}
    stats = {
        "enabled": True, "scope": scope, "max_hag": max_hag, "target_class": target,
        "target_client_class": int(spec["ptc_class_mapping"].get(target, target)),
        "source_classes": list(sources), "absorbed": int(mask.sum()),
        "absorbed_by_source": per_source,
        "kept_to_preserve_thinning": int(np.count_nonzero(source_mask & near_terrain & protected)),
    }

    detail = " | ".join(f"class {c} {class_name(c)}: {per_source[c]:,}" for c in sources)
    print(f"        [Rule G] Terrain-hugging speckle at or below {max_hag:g} m HAG -> internal {target} "
          f"({class_name(target)}), delivered as {_delivered_label(spec, target)}, "
          f"scope={scope}: {int(mask.sum()):,} pts")
    print(f"                  {detail}")
    if stats["kept_to_preserve_thinning"]:
        print(f"                  {stats['kept_to_preserve_thinning']:,} pts were left alone - "
              "they were unclassified by the Rule D sieve, which this rule must not undo")

    # Density of what is being absorbed: speckle is a few points per cell, a water body is not.
    if mask.any():
        inverse, _ = _grid_cells(x[mask], y[mask], 2.0)
        counts = np.bincount(inverse)
        counts = counts[counts > 0]
        stats["absorbed_cells_2m"] = int(len(counts))
        stats["absorbed_max_points_per_cell"] = int(counts.max())
        water = per_source.get(WATER_CODE, 0)
        stats["water_absorbed"] = water
        print(f"                  footprint: {len(counts):,} cells of 2 m, median "
              f"{np.median(counts):.0f} / max {int(counts.max()):,} pts per cell")
        if water:
            print(f"                  [!] {water:,} of them are Water: if that water has to stay "
                  "visible, set ground_refinement.enabled=false or lower\n"
                  "                      ground_refinement.max_hag.")

    stats["changed"] = _assign(classes, mask, target)
    provenance[mask] = ORIGIN_GROUND_HOMOGENIZATION
    print(f"        [Rule G] Reclassified {int(mask.sum()):,} pts -> Class {target} "
          f"({class_name(target)})")
    return stats


def rule_d_ground_subsample(classes, spec, defaulted_mask=None, provenance=None):
    """Unclassify a share of the ground pool (seeded, so the deliverable is reproducible)."""
    g = spec["ground_subsampling"]
    if not bool(g["enabled"]):
        print("        [Rule D] Disabled by contract (ground_subsampling.enabled=false)")
        return {"dropped": 0, "pool": 0}
    pct = float(g["target_percentage"])
    seed = int(g["random_seed"])
    pool = str(g["source_pool"])

    ground_mask = classes == 2
    excluded = 0
    if pool == "spurious_veg_excluded" and defaulted_mask is not None:
        excluded = int(np.count_nonzero(ground_mask & defaulted_mask))
        ground_mask = ground_mask & ~defaulted_mask
    ground_idx = np.flatnonzero(ground_mask)
    n_drop = int(len(ground_idx) * (pct / 100.0))
    if n_drop:
        # Keyed on the point, not on the pool: keep the n_drop ground points with
        # the lowest per-row uniform.  The count stays exactly pct% of the pool.
        u = _stable_uniform(seed, ground_idx)
        picked = ground_idx[np.argpartition(u, n_drop - 1)[:n_drop]]
        classes[picked] = 1
        if provenance is not None:
            provenance[picked] = ORIGIN_GROUND_THINNING
    note = f"; {excluded:,} Rule A defaulted pts excluded by source_pool={pool}" if excluded else ""
    print(f"        [Rule D] source_pool={pool}: pool {len(ground_idx):,} pts{note} -> "
          f"unclassified {n_drop:,} ({pct:g}%, seed {seed})")
    return {"dropped": n_drop, "pool": int(len(ground_idx)), "excluded": excluded}


def rule_e_remap(classes, spec):
    """Map internal codes to client .ptc IDs and apply the unmapped class policy."""
    mapping = spec["ptc_class_mapping"]
    policy = spec["unmapped_class_policy"]
    present = np.unique(classes)
    unmapped = [int(c) for c in present if int(c) not in mapping]

    final = classes.copy()
    for src, dst in mapping.items():
        final[classes == src] = dst

    counts = {c: int(np.count_nonzero(classes == c)) for c in unmapped}
    if not unmapped:
        print(f"        [Rule E] Mapped {len(mapping)} contract codes; every code present in the "
              "cloud is covered by the schema (unmapped_class_policy unused)")
    elif policy == "unclassify":
        stats_changed = _assign(final, np.isin(classes, unmapped), 1)
        print(f"        [Rule E] Unmapped codes {[(c, class_name(c), counts[c]) for c in unmapped]} "
              f"-> Class 1 ({stats_changed:,} pts, policy=unclassify)")
    else:
        detail = ", ".join(f"{c} ({class_name(c)}): {counts[c]:,}" for c in unmapped)
        print(f"        [Rule E] Unmapped codes preserved (policy=preserve): {detail}")

    return final, {"unmapped": unmapped, "unmapped_counts": counts, "policy": policy}


def enforce_output_class_set(final_classes, spec, provenance):
    """Rule H: the delivered file may only carry classes the client asked for.

    Rule E targets whatever code the contract names, and a .ptc name match can point at a class
    outside the client's deliverable schema - the client's 10-class set has no Water, Railway or
    Street class at all. Nothing is dropped silently here: every offending code is counted,
    routed (outlier origins to the noise container, everything else to the fallback container)
    and reported, so the file is guaranteed to hold exactly the client's target set.
    """
    block = spec["output_class_set"]
    allowed = [int(c) for c in block["allowed"]]
    stats = {"enforce": bool(block["enforce"]), "allowed": allowed, "coerced": {}}
    present = [int(c) for c in np.unique(final_classes)]
    stray = [c for c in present if c not in allowed]
    stats["delivered_before"] = present
    stats["out_of_spec_before"] = int(np.count_nonzero(~np.isin(final_classes, allowed)))
    if not bool(block["enforce"]):
        print(f"        [Rule H] Disabled by contract (output_class_set.enforce=false); "
              f"delivered codes {present} (target set {allowed})")
        return stats
    if stray:
        fallback = int(block["fallback_class"])
        noise_fallback = int(block["noise_fallback_class"])
        noise_origin = np.isin(provenance, (ORIGIN_LOW_NOISE, ORIGIN_HIGH_NOISE))
        names = display_names(spec)
        print(f"        [Rule H] Codes outside the client target set {allowed} are routed to "
              f"Class {fallback} / Class {noise_fallback} (noise origins):")
        for code in stray:
            sel = final_classes == code
            n_other = int(np.count_nonzero(sel & ~noise_origin))
            n_noise = int(np.count_nonzero(sel & noise_origin))
            routes = []
            if n_other:
                routes.append(f"{n_other:,} -> Class {fallback} "
                              f"({names.get(fallback, class_name(fallback))})")
            if n_noise:
                routes.append(f"{n_noise:,} -> Class {noise_fallback} "
                              f"({names.get(noise_fallback, class_name(noise_fallback))})")
            print(f"                  class {code} {names.get(code, class_name(code))}: "
                  f"{int(np.count_nonzero(sel)):,} pts | " + " | ".join(routes))
            stats["coerced"][str(code)] = {"count": int(np.count_nonzero(sel)),
                                           "to_fallback": n_other, "to_noise": n_noise,
                                           "fallback": fallback, "noise_fallback": noise_fallback}
        final_classes[~np.isin(final_classes, allowed) & noise_origin] = noise_fallback
        final_classes[~np.isin(final_classes, allowed)] = fallback
    present = [int(c) for c in np.unique(final_classes)]
    stats["delivered"] = present
    if not np.isin(final_classes, allowed).all():   # pragma: no cover - defensive
        raise SystemExit("[X] Rule H could not force every point into the client target set")
    print(f"        [Rule H] Delivered class set {present} - inside the client's target set "
          f"{allowed}")
    return stats


# ------------------------------------------------------------------------------ verification
def verify_output(path: str, las_in, final_classes: np.ndarray, allowed=None) -> None:
    """Round-trip and bit-identity checks on the delivered artefact."""
    input_dims = set(las_in.point_format.dimension_names)
    with laspy.open(path) as f:
        written = f.header.point_count
        fmt_id = f.header.point_format.id
        out_dims = set(f.header.point_format.dimension_names)

    if written != len(las_in.points):
        raise SystemExit(f"[X] Point count mismatch: wrote {written:,}, expected {len(las_in.points):,}")
    if fmt_id != las_in.header.point_format.id:
        raise SystemExit(f"[X] Point format changed: {las_in.header.point_format.id} -> {fmt_id}")
    if input_dims - out_dims:
        raise SystemExit(f"[X] Dimensions lost on write: {sorted(input_dims - out_dims)}")

    check = laspy.read(path)
    if not np.array_equal(np.asarray(check.classification), final_classes.astype(np.uint8)):
        raise SystemExit("[X] Verification failed: classification differs after round-trip")

    compared, mismatches = [], []
    for name in sorted(set(check.point_format.dimension_names) - {"classification"}):
        if np.array_equal(np.asarray(check[name]), np.asarray(las_in[name])):
            compared.append(name)
        else:
            mismatches.append(name)
    if mismatches:
        raise SystemExit(f"[X] Non-classification dimensions altered: {mismatches}")

    print(f"        [OK] Verified {path}: {written:,} pts, point format {fmt_id}, "
          f"{len(out_dims)} dimensions preserved,\n"
          f"             classification round-trips exactly and all {len(compared)} "
          "non-classification dimensions are bit-identical")

    if allowed is not None:
        present = [int(c) for c in np.unique(final_classes)]
        stray = [c for c in present if c not in allowed]
        if stray:
            raise SystemExit(f"[X] Verification failed: delivered classes {stray} are outside "
                             f"the client target set {list(allowed)}")
        print(f"             delivered class set {present} is inside the client's target set "
              f"{list(allowed)}")


# ------------------------------------------------------------------------------------ driver
def run_step3_rule_engine(
    input_ai_laz: str = "ai_predicted.laz",
    output_final_laz: str = "output_classified.laz",
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    seed: int = None,
    ground_unclass_pct: float = None,
    unmapped_policy: str = None,
    use_spatial_hag: bool = None,
    ground_source_pool: str = None,
):
    """Run rules A-H from the contract spec and write the final deliverable."""
    print(f"[Step 3] Executing Contract-Driven Rule Engine on: {input_ai_laz}")
    start = time.time()

    spec = load_contract_spec(config_path)
    resolved = spec["_config_path"]
    print(f"        Contract '{spec.get('contract_name', 'unnamed')}' loaded from "
          f"{resolved} (md5 {_md5(resolved)})")
    if spec["_extension_defaults_applied"]:
        print(f"        Contract supplied {len(spec['_extension_defaults_applied'])} fewer keys than "
              f"this engine knows; defaults applied: {spec['_extension_defaults_applied']}")
    # Runtime overrides (CLI / API) take precedence over the contract.
    if seed is not None:
        spec["ground_subsampling"]["random_seed"] = int(seed)
    if ground_unclass_pct is not None:
        spec["ground_subsampling"]["target_percentage"] = float(ground_unclass_pct)
    if ground_source_pool is not None:
        spec["ground_subsampling"]["source_pool"] = str(ground_source_pool).lower()
    if unmapped_policy is not None:
        spec["unmapped_class_policy"] = str(unmapped_policy).lower()
    if use_spatial_hag is not None:
        spec["outlier_detection"]["use_spatial_hag"] = bool(use_spatial_hag)

    las = laspy.read(input_ai_laz)
    total_pts = len(las.points)
    dims = set(las.point_format.dimension_names)
    if "classification" not in dims:
        raise SystemExit(f"[X] {input_ai_laz} has no classification dimension")
    if "HeightAboveGround" not in dims:
        raise SystemExit("[X] Missing required 'HeightAboveGround' dimension")
    z_rel = np.asarray(las.HeightAboveGround, dtype=np.float64)
    if not np.isfinite(z_rel).all():
        bad = int((~np.isfinite(z_rel)).sum())
        print(f"        [!] {bad:,} non-finite HeightAboveGround values coerced to 0 m")
        z_rel = np.where(np.isfinite(z_rel), z_rel, 0.0)
    report_hag_anchor(las, z_rel)
    classes = np.asarray(las.classification, dtype=np.uint8).copy()
    input_classes = classes.copy()
    print(f"        Loaded {total_pts:,} points (point format {las.header.point_format.id}, "
          f"{las.header.version} vers, {len(dims)} dimensions) in {time.time() - start:.2f} s")

    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z_raw = np.asarray(las.z, dtype=np.float64)
    intensity = np.asarray(las.intensity, dtype=np.float64) if "intensity" in dims else None
    rn = np.asarray(las.return_number) if "return_number" in dims else None
    nr = np.asarray(las.number_of_returns) if "number_of_returns" in dims else None

    # Each rule is timed: the console output is the only performance evidence a client sees, and
    # the covariance search in Rule F dominates the run.
    timings = {}

    def timed(name, fn, *args, **kwargs):
        t0 = time.time()
        out = fn(*args, **kwargs)
        timings[name] = time.time() - t0
        return out

    provenance = np.full(total_pts, ORIGIN_AI, dtype=np.uint8)
    rule_a = timed("A", rule_a_vegetation_split, classes, z_rel, spec, provenance)
    # Rule N straight after A: the deliverable has no Bridge class, so no later rule may read it.
    rule_n = timed("N", rule_n_bridge_disposal, classes, z_rel, x, y, spec, provenance)
    timed("B", rule_b_building_recovery, classes, z_rel, x, y, spec, intensity=intensity,
          return_number=rn, number_of_returns=nr, provenance=provenance)
    # Rule F first: a recovered wire or pole is then covered by Rule C's preserve_classes, so
    # the noise rules cannot throw away the very structures this rule just found. Rule F itself
    # sequences poles -> isolated noise -> conductors -> bridge -> pole columns (see its
    # docstring), so isolated high returns are already noise before any wire is claimed.
    rule_f = timed("F", rule_f_network_extraction, classes, z_rel, x, y, z_raw, spec, provenance)
    # Rule I straight after F: it judges F's finished wire set, and it must hand the crowns back
    # to vegetation before Rule C, whose preserve_classes would otherwise shield the false wires.
    # The utility mask is frozen first, from F's finished poles and wires: Rule I may never
    # modify a pole, a confirmed wire or anything inside the 2D wire corridor.
    # Rule L first: a pink patch it rejects above roof height is handed to Rule I as
    # Unclassified, so the crown test below decides whether it is canopy.
    rule_l = timed("L", rule_l_building_validation, classes, z_rel, x, y, z_raw, spec,
                   provenance)
    # Rule R after L: L has re-filed the false pink, so what R adds is a whole roof, not bleed.
    rule_r, structure_mask = timed("R", rule_r_roof_consolidation, classes, z_rel, x, y, z_raw,
                                   spec, provenance)
    low_wire = timed("W", purge_low_wire, classes, z_rel, x, y, z_raw, spec, provenance)
    # Rule T before the utility mask: it traces the network from the poles that stand in the open
    # and hands the poles and wire it reaches to the mask, so Rules I and J cannot lose them.
    rule_t = timed("T", rule_t_network_trace, classes, z_rel, x, y, z_raw, spec, provenance)
    immutable_utility_mask, utility_stats, confirmed_wire = timed(
        "U", build_immutable_utility_mask, classes, x, y, z_raw, spec, return_valid=True,
        z_rel=z_rel, structure=structure_mask, traced=rule_t)
    rule_i = timed("I", rule_i_canopy_homogenization, classes, z_rel, x, y, z_raw, spec,
                   provenance, immutable_utility_mask, wire_idx=confirmed_wire)
    rule_i["utility_protection"] = utility_stats
    rule_i["low_wire"] = low_wire
    # Rule J after I: the canopy around each pole is vegetation by now, so the envelope it cuts
    # embedded poles at is the real one. It is a pole rule - the only one allowed past the
    # utility mask, and only to hand the submerged part of a column to vegetation.
    rule_j = timed("J", rule_j_exposed_pole_truncation, classes, z_rel, x, y, z_raw, spec,
                   provenance, wire_idx=confirmed_wire, traced_poles=rule_t["pole_idx"])
    # Rule K after J: the support nodes come from the confirmed wire, and the cutoff it applies
    # is J's vegetation envelope, measured on the canopy J has just finished.
    rule_k = timed("K", rule_k_wire_guided_pole_recovery, classes, z_rel, x, y, z_raw, spec,
                   provenance, wire_idx=confirmed_wire)
    # Rule X before C: wire left as a small clump in a crown is foliage; once it is vegetation,
    # Rule V no longer has to keep the points around it Unclassified.
    rule_x = timed("X", purge_wire_blobs, classes, z_rel, x, y, z_raw, spec, provenance)
    timed("C", rule_c_outliers, classes, z_rel, x, y, z_raw, spec, provenance)
    # Rule V after C: outliers are noise by now, so only the real low layer is left to fill.
    rule_v = timed("V", rule_v_low_vegetation_fill, classes, z_rel, x, y, z_raw, spec,
                   provenance)
    speckle_scope = spec["ground_refinement"]["scope"]
    rule_g = None
    if speckle_scope == "before_thinning":
        rule_g = timed("G", rule_g_ground_refinement, classes, z_rel, x, y, spec, provenance)
    timed("D", rule_d_ground_subsample, classes, spec, rule_a["defaulted_mask"], provenance)
    if rule_g is None:
        rule_g = timed("G", rule_g_ground_refinement, classes, z_rel, x, y, spec, provenance,
                       after_thinning=True)
    final_classes, remap_stats = timed("E", rule_e_remap, classes, spec)
    output_set = timed("H", enforce_output_class_set, final_classes, spec, provenance)
    print("        Rule timings: " + " | ".join(f"{k} {v:.2f} s" for k, v in timings.items()))
    breakdown = class_zero_breakdown(final_classes, spec, provenance, input_classes)
    print_class_zero_breakdown(breakdown)

    las.classification = final_classes.astype(np.uint8)
    las.write(output_final_laz)
    verify_output(output_final_laz, las, final_classes,
                  allowed=output_set["allowed"] if output_set["enforce"] else None)

    elapsed = time.time() - start
    print(f"        Post-processing completed in {elapsed:.2f} seconds. "
          f"Output md5 {_md5(output_final_laz)}")
    print_summary(final_classes, total_pts, spec)
    print_change_table(input_classes, final_classes, spec)
    print(f"[Step 3] Done -> Final Output Saved: {output_final_laz}\n")
    return {"seconds": elapsed, "points": total_pts, "output": output_final_laz,
            "contract": spec.get("contract_name"), "remap": remap_stats,
            "rule_timings": {k: round(v, 3) for k, v in timings.items()},
            "class_0_breakdown": breakdown,
            "network_extraction": rule_f, "network_trace": {
                # The debug arrays Rule T keeps for a cross-section (heads, trusted, seed, spans)
                # start with "_" and are dropped here so they never reach the contract or the report.
                k: v for k, v in rule_t.items()
                if k not in ("pole_idx", "wire_idx", "protected_poles") and not k.startswith("_")},
            "canopy_homogenization": rule_i,
            "low_vegetation_fill": rule_v,
            "wire_blob_purge": rule_x,
            "pole_truncation": rule_j, "pole_recovery": rule_k,
            "bridge_disposal": rule_n, "roof_consolidation": rule_r,
            "building_validation": rule_l,
            "ground_refinement": rule_g,
            "output_class_set": output_set,
            "ground_subsample": spec["ground_subsampling"]["target_percentage"]}


def class_zero_breakdown(final_classes: np.ndarray, spec: dict, provenance: np.ndarray,
                         input_classes: np.ndarray = None) -> dict:
    """Split the client's unassigned container by origin.

    Several rules can deliver into the same client class (with a TerraScan .ptc that class is
    0): ground thinning from Rule D, low and high outliers from Rule C, the model's own
    unclassified points and - when the deliverable schema has no Water class - the water
    outside the ground band. A client auditing the file must be able to tell them apart, so each
    origin is counted separately, and the closing bucket is the remainder rather than an
    assumption, which keeps the parts adding up to the container total exactly.
    """
    mapping = spec["ptc_class_mapping"]
    container = int(mapping.get(1, 1))          # whatever internal Unclassified maps to
    in_container = final_classes == container
    counts = {origin: int(np.count_nonzero(in_container & (provenance == origin)))
              for origin in ORIGIN_LABELS}
    if input_classes is None:
        water = 0
    else:
        water = int(np.count_nonzero(in_container & (provenance == ORIGIN_AI)
                                     & (input_classes == WATER_CODE)))
    parts = {
        "class_0_unclassified_ground_count": counts[ORIGIN_GROUND_THINNING],
        "class_0_high_noise_count": counts[ORIGIN_HIGH_NOISE],
        "class_0_low_noise_count": counts[ORIGIN_LOW_NOISE],
        "class_0_water_count": water,
        "class_0_ai_unclassified_count": counts[ORIGIN_AI] - water,
        "class_0_spurious_veg_count": counts[ORIGIN_SPURIOUS_VEG],
    }
    parts["class_0_other_count"] = int(in_container.sum()) - sum(parts.values())
    return {
        "container_class": container,
        "container_total": int(in_container.sum()),
        **parts,
        "component_sum": sum(parts.values()),
        "origins": {ORIGIN_LABELS[o]: c for o, c in counts.items() if c},
        "noise_destinations": {
            "internal_1_unclassified": container,
            "internal_7_low_noise": int(mapping.get(7, 7)),
            "internal_18_high_noise": int(mapping.get(18, 18)),
        },
        "container_is_literal_class_0": container == 0,
    }


def print_class_zero_breakdown(breakdown: dict) -> None:
    container = breakdown["container_class"]
    print(f"        [Class {container}] unassigned container: "
          f"{breakdown['container_total']:,} pts (internal Unclassified maps here)")
    rows = (("class_0_unclassified_ground_count", "Rule D ground thinning",
             breakdown["class_0_unclassified_ground_count"]),
            ("class_0_high_noise_count", "Rule C high noise",
             breakdown["class_0_high_noise_count"]),
            ("class_0_low_noise_count", "Rule C low noise",
             breakdown["class_0_low_noise_count"]),
            ("class_0_ai_unclassified_count", "AI unclassified",
             breakdown["class_0_ai_unclassified_count"]),
            ("class_0_water_count", "Water (no client class)",
             breakdown["class_0_water_count"]),
            ("class_0_spurious_veg_count", "Rule A spurious vegetation",
             breakdown["class_0_spurious_veg_count"]),
            ("class_0_other_count", "other rules", breakdown["class_0_other_count"]))
    for key, label, value in rows:
        print(f"                  {label:<26} {value:>11,}  {key}")
    destination = breakdown["noise_destinations"]
    if not breakdown["class_0_low_noise_count"] and destination["internal_7_low_noise"] != container:
        print(f"                  note: low noise is delivered as client class "
              f"{destination['internal_7_low_noise']} per the contract mapping, so it is not "
              "counted here")
    if not breakdown["class_0_high_noise_count"] and destination["internal_18_high_noise"] != container:
        print(f"                  note: high noise is delivered as client class "
              f"{destination['internal_18_high_noise']} per the contract mapping")


def display_names(spec: dict) -> dict:
    """Names for the delivered codes: the client schema when the contract carries one.

    A client .ptc renumbers classes (e.g. client 6 = Water, where our internal 6 = Building),
    so reporting the delivered codes with internal names would mislabel the file. The resolved
    contract may therefore carry a ``class_names`` block, used for display only.
    """
    if not spec.get("class_names"):
        return dict(CLASS_NAMES)
    names = dict(CLASS_NAMES)
    for key, value in spec["class_names"].items():
        try:
            names[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return names


def print_summary(final_classes: np.ndarray, total_pts: int, spec: dict) -> None:
    names = display_names(spec)
    client_schema = bool(spec.get("class_names"))
    codes, counts = np.unique(final_classes, return_counts=True)
    print("\n        ================ FINAL DELIVERABLE SUMMARY ================")
    if client_schema:
        print("        (class names from the client .ptc schema)")
    print(f"        {'Class':>5}  {'Name':<18} {'Points':>12}  {'Share':>7}")
    print("        " + "-" * 50)
    for c, n in zip(codes, counts):
        print(f"        {int(c):>5}  {names.get(int(c), class_name(c)):<18} "
              f"{int(n):>12,}  {n / total_pts * 100:>6.2f}%")
    print("        " + "-" * 50)
    print(f"        {'':>5}  {'TOTAL':<18} {total_pts:>12,}  100.00%")
    print("        ===========================================================")
    block = spec.get("output_class_set")
    if block and block.get("enforce"):
        print(f"        Target class set  : {list(block['allowed'])} - every delivered code is "
              "inside it (Rule H)")


def print_change_table(before: np.ndarray, after: np.ndarray, spec: dict) -> None:
    """Before/after counts.

    With a client schema the two arrays live in different code spaces (``before`` internal,
    ``after`` client), so the AI column is folded through the contract mapping first: the row
    then reads "what the model's own classes would have delivered" and the delta is exactly what
    the rules changed. Without a client schema both sides are the same space, shown as they are.
    """
    names = display_names(spec)
    print("\n        --- Input (AI) vs Output (delivered) ---")
    if spec.get("class_names"):
        mapping = spec["ptc_class_mapping"]
        folded, sources = {}, {}
        for c in np.unique(before):
            target = int(mapping.get(int(c), int(c)))
            folded[target] = folded.get(target, 0) + int(np.count_nonzero(before == c))
            sources.setdefault(target, []).append(int(c))
        print("        (the AI column is folded through the contract mapping, so both columns "
              "are client codes)")
        print(f"        {'Client':>6}  {'Name':<22} {'AI (mapped)':>12} {'Final':>12}  "
              f"{'Delta':>12}  From internal")
        for c in sorted(set(folded) | {int(v) for v in np.unique(after)}):
            b = folded.get(c, 0)
            a = int(np.count_nonzero(after == c))
            src = ", ".join(f"{s} {CLASS_NAMES.get(s, '?')}" for s in sources.get(c, []))
            print(f"        {c:>6}  {names.get(c, class_name(c)):<22} {b:>12,} {a:>12,}  "
                  f"{a - b:>+12,}  {src}")
        return

    print(f"        {'Class':>5}  {'Name':<18} {'AI':>12} {'Final':>12}  {'Delta':>12}")
    for c in sorted(set(np.unique(before).tolist()) | set(np.unique(after).tolist())):
        b = int(np.count_nonzero(before == c))
        a = int(np.count_nonzero(after == c))
        print(f"        {int(c):>5}  {class_name(c):<18} {b:>12,} {a:>12,}  {a - b:>+12,}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Step 3: contract-driven rule engine, spatial HAG cleanup, .ptc export")
    ap.add_argument("input", nargs="?", default="ai_predicted.laz", help="Step 2 output LAZ")
    ap.add_argument("output", nargs="?", default="output_classified.laz", help="final deliverable")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="client contract spec (JSON)")
    ap.add_argument("--seed", type=int, help="override ground_subsampling.random_seed")
    ap.add_argument("--ground-unclass-pct", type=float,
                    help="override ground_subsampling.target_percentage")
    ap.add_argument("--ground-source-pool", choices=SOURCE_POOLS,
                    help="override ground_subsampling.source_pool")
    ap.add_argument("--unmapped-policy", choices=UNMAPPED_POLICIES,
                    help="override unmapped_class_policy")
    ap.add_argument("--spatial-hag", dest="use_spatial_hag", action="store_true", default=None,
                    help="force the local-window HAG detector on")
    ap.add_argument("--no-spatial-hag", dest="use_spatial_hag", action="store_false",
                    help="fall back to the contract's absolute HAG bounds only")
    ap.add_argument("--print-spec", action="store_true", help="print the effective contract")
    a = ap.parse_args(argv)

    if a.print_spec:
        spec = load_contract_spec(a.config)
        spec.pop("_extension_defaults_applied", None)
        spec.pop("_config_path", None)
        print(json.dumps(spec, indent=2, sort_keys=True))
        return 0

    run_step3_rule_engine(
        a.input, a.output, a.config,
        seed=a.seed, ground_unclass_pct=a.ground_unclass_pct,
        unmapped_policy=a.unmapped_policy, use_spatial_hag=a.use_spatial_hag,
        ground_source_pool=a.ground_source_pool,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
