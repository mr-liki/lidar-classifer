// LNTBuilder.h - public interface of the .lnt offline converter.
//
// Declarations only. The implementation lives in LNTBuilderImpl.h, which is
// included by exactly ONE translation unit:
//
//   header-only (the default, matches the plain one-command build)
//       main.cpp includes LNTBuilder.h then LNTBuilderImpl.h
//       g++ -O3 -std=c++17 main.cpp -o main
//
//   split build (for a larger project that compiles the builder once)
//       g++ -O3 -std=c++17 -DLNT_EXTERN_BUILDER main.cpp src/LNTBuilder.cpp -o main
//
// Keeping the body in a separate impl header is what makes both work with no
// duplicated code and no ODR hazard: the definitions exist in exactly one TU
// either way.
//
// Pipeline (four stages, one pass over the input plus one over the sorted keys):
//
//   1. BOUNDARY SCAN     parallel bounds reduce, then force a cubic envelope
//   2. MORTON KEYS       parallel quantise to 21 bits/axis + 64-bit interleave
//   3. RADIX + OCTREE    parallel O(N) LSD radix sort, then prefix bucketing
//   4. WRITE             sequential LNTPoint chunks, directory at index_offset

#ifndef LNT_BUILDER_H
#define LNT_BUILDER_H

#include <cstddef>
#include <cstdint>
#include <string>

#include "LNTTypes.h"

namespace lnt {

struct LNTBuildParams {
    // A node stops splitting once it holds at most this many points. This is
    // the primary subdivision control.
    uint64_t max_points_per_node = 60000;

    // Optional precision-driven subdivision. The 16-bit local quantisation of
    // LNTPoint resolves 2*half_size/65535 metres, so a node that is 1200 m
    // across can only place a point to ~18 mm. Setting this to a survey
    // precision (e.g. 0.002) keeps splitting until the cell is fine enough,
    // bounded below by min_points_per_node so a sparse region is never
    // shredded into single-point nodes. 0 disables it (capacity-only tree).
    double target_resolution = 0.0;

    // Floor for target_resolution: never split a node holding fewer than this.
    uint64_t min_points_per_node = 4096;

    // 0 => std::thread::hardware_concurrency().
    unsigned threads = 0;

    // Force the envelope cubic. Required by the octree (one `half_size` per
    // node must describe all three axes) - see Bounds3d::make_cube(). Off is
    // offered only so a test can show what breaks without it.
    bool cube_bounds = true;

    // Interior (non-leaf) nodes carry a strided LOD chunk instead of the full
    // descendant set, capped at this many points.
    uint64_t parent_lod_target = 60000;

    bool verbose = false;
};

struct LNTBuildStats {
    uint64_t input_points   = 0;
    uint64_t leaf_points    = 0;   // points held by leaves; == input_points
    uint64_t lod_points     = 0;   // extra points written for interior nodes
    uint64_t written_points = 0;   // leaf_points + lod_points
    uint64_t node_count     = 0;
    uint64_t leaf_count     = 0;
    uint64_t max_depth      = 0;
    uint64_t file_bytes     = 0;
    uint64_t index_offset   = 0;

    unsigned threads = 0;
    Bounds3d bounds;               // the CUBIC envelope actually used

    // Stage timings, seconds.
    double scan_s = 0.0;
    double morton_s = 0.0;
    double sort_s = 0.0;
    double tree_s = 0.0;
    double write_s = 0.0;
    double total_s = 0.0;

    // Honest precision reporting. Both figures are measured while writing, not
    // assumed from a formula.
    //   max_*        worst over EVERY node. The root always spans the whole
    //                cube, so this value is dominated by the root's coarse cell
    //                and cannot show whether deeper subdivision helped.
    //   max_leaf_*   worst over LEAF nodes only: the precision actually achieved
    //                for full-detail geometry, and the figure to compare
    //                between two builds.
    double max_cell_step = 0.0;
    double max_recon_error = 0.0;
    double max_leaf_cell_step = 0.0;
    double max_leaf_recon_error = 0.0;

    // The sort is stable by construction; recorded so a test can assert it
    // rather than trust it.
    bool stable_sort = false;
    bool ok = false;
    std::string error;
};

class LNTBuilder {
public:
    // Convert `n` raw points into a single .lnt file at `out_path`.
    // Never throws: a failure is reported through stats.ok / stats.error.
    static LNTBuildStats build(const RawPoint* pts, size_t n,
                               const char* out_path,
                               const LNTBuildParams& params = LNTBuildParams());
};

}  // namespace lnt

#endif  // LNT_BUILDER_H
