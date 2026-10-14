// LNTBuilderImpl.h - the .lnt pipeline implementation.
//
// Include from exactly ONE translation unit (see LNTBuilder.h).

#ifndef LNT_BUILDER_IMPL_H
#define LNT_BUILDER_IMPL_H

// Linkage switch. This header holds the ONLY definition of LNTBuilder::build,
// so the specifier has to change with the role:
//
//   consumer (header-only build): `inline`, so the definition is emitted in
//       whichever TU uses it - and main.cpp is the only one.
//   builder TU (split build): nothing. An `inline` definition in a TU that
//       does not itself call the function is DISCARDED, which is why the first
//       split build linked with an undefined reference: src/LNTBuilder.cpp
//       compiled to an object file with no LNTBuilder::build in it at all.
#ifdef LNT_BUILDER_TU
#  define LNT_BUILD_FN
#else
#  define LNT_BUILD_FN inline
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "LNTBoundaryScanner.h"
#include "LNTBuilder.h"
#include "LNTFormat.h"
#include "LNTMath.h"
#include "LNTTypes.h"

namespace lnt {
namespace detail {

// ==========================================================================
// Stage 2 - Morton keys, computed in parallel over contiguous chunks
// ==========================================================================
inline void compute_morton_keys(const RawPoint* pts, const Bounds3d& cube,
                                uint64_t* keys, uint32_t* ids, size_t n,
                                unsigned T) {
    const unsigned workers = static_cast<unsigned>(
        std::min<size_t>(T, std::max<size_t>(n, 1)));
    const size_t chunk = (n + workers - 1) / workers;

    // The cubic envelope is captured by value into each chunk loop: the
    // quantiser needs min AND max per axis, and reading them from a shared
    // struct inside the hot loop would defeat vectorisation.
    const double x0 = cube.min[0], x1 = cube.max[0];
    const double y0 = cube.min[1], y1 = cube.max[1];
    const double z0 = cube.min[2], z1 = cube.max[2];

    run_workers(workers, [&](unsigned t) {
        const size_t begin = static_cast<size_t>(t) * chunk;
        const size_t end = std::min(n, begin + chunk);
        for (size_t i = begin; i < end; ++i) {
            const RawPoint& p = pts[i];
            const uint32_t qx = quantize_to_21bit(p.x, x0, x1);
            const uint32_t qy = quantize_to_21bit(p.y, y0, y1);
            const uint32_t qz = quantize_to_21bit(p.z, z0, z1);
            keys[i] = encode_morton_64(qx, qy, qz);
            ids[i] = static_cast<uint32_t>(i);
        }
    });
}

// ==========================================================================
// Stage 3a - parallel stable LSD radix sort, O(N)
// ==========================================================================
// 8 passes of one byte, least significant first. Stability is the load-bearing
// property: pass k must preserve the ordering established by passes 0..k-1,
// otherwise the final order is not Morton order at all.
//
// Per pass:
//   1. each worker histograms its own contiguous chunk           (no sharing)
//   2. one global exclusive scan over (digit-major, worker-minor)
//   3. each worker scatters its chunk into the destination
//
// Stability argument: within a digit d the destinations are laid out as
// worker 0's run, then worker 1's, and so on. Each worker scatters its own
// elements in ascending source order, and chunk t holds lower source indices
// than chunk t+1. So inside every digit the relative order of the source is
// reproduced exactly - which is all LSD stability requires.
class RadixSort {
public:
    static void sort(uint64_t* keys, uint32_t* ids, size_t n, unsigned T) {
        if (n < 2) return;
        const unsigned workers = static_cast<unsigned>(
            std::min<size_t>(T, std::max<size_t>(n, 1)));
        const size_t chunk = (n + workers - 1) / workers;

        std::vector<uint64_t> dk(n);
        std::vector<uint32_t> di(n);
        std::vector<uint64_t> hist(static_cast<size_t>(workers) * 256, 0);
        std::vector<uint64_t> offs(static_cast<size_t>(workers) * 256, 0);

        uint64_t* src_k = keys;
        uint32_t* src_i = ids;
        uint64_t* dst_k = dk.data();
        uint32_t* dst_i = di.data();

        for (int pass = 0; pass < 8; ++pass) {
            const int shift = 8 * pass;

            std::fill(hist.begin(), hist.end(), uint64_t(0));
            run_workers(workers, [&](unsigned t) {
                const size_t begin = static_cast<size_t>(t) * chunk;
                const size_t end = std::min(n, begin + chunk);
                uint64_t* h = &hist[static_cast<size_t>(t) * 256];
                for (size_t i = begin; i < end; ++i)
                    ++h[(src_k[i] >> shift) & 0xFFu];
            });

            // Exclusive scan, digit-major then worker-minor: this is what
            // stitches the per-worker runs into one stable global order.
            uint64_t running = 0;
            for (int d = 0; d < 256; ++d) {
                for (unsigned t = 0; t < workers; ++t) {
                    const size_t at = static_cast<size_t>(t) * 256 + d;
                    offs[at] = running;
                    running += hist[at];
                }
            }

            run_workers(workers, [&](unsigned t) {
                const size_t begin = static_cast<size_t>(t) * chunk;
                const size_t end = std::min(n, begin + chunk);
                uint64_t* o = &offs[static_cast<size_t>(t) * 256];
                for (size_t i = begin; i < end; ++i) {
                    const uint64_t k = src_k[i];
                    const uint64_t pos = o[(k >> shift) & 0xFFu]++;
                    dst_k[pos] = k;
                    dst_i[pos] = src_i[i];
                }
            });

            std::swap(src_k, dst_k);
            std::swap(src_i, dst_i);
        }

        // An even number of swaps lands the sorted data back in the caller's
        // buffers, so normally no copy happens at all. The guard keeps that a
        // property of the algorithm rather than of "8 happen to be even": if
        // the pass count ever changes to an odd number, this copies once.
        if (src_k != keys) {
            std::memcpy(keys, src_k, n * sizeof(uint64_t));
            std::memcpy(ids, src_i, n * sizeof(uint32_t));
        }
    }
};

// ==========================================================================
// Stage 3b - octree bucketing over the sorted Morton keys
// ==========================================================================
struct NodeRec {
    uint64_t node_id = 0;
    uint64_t begin = 0;
    uint64_t end = 0;
    double   center[3] = {0, 0, 0};
    double   half_size = 0.0;
    uint8_t  level = 0;
    uint8_t  children_mask = 0;
    bool     is_leaf = true;

    // Filled while writing.
    uint64_t byte_offset = 0;
    uint64_t point_count = 0;
    uint64_t byte_length = 0;
};

class OctreeBuilder {
public:
    OctreeBuilder(const uint64_t* keys, const Bounds3d& cube,
                  const LNTBuildParams& p)
        : keys_(keys), params_(p) {
        double c[3];
        cube.center(c);
        root_center_[0] = c[0];
        root_center_[1] = c[1];
        root_center_[2] = c[2];
        root_half_ = 0.5 * cube.max_extent();
    }

    void build() {
        nodes_.reserve(1024);
        // Pre-order: a node is appended BEFORE its children, so parents - and
        // therefore their coarse LOD chunks - land earlier in the file and a
        // reader can paint a low-detail view after reading only the head of it.
        //
        // The root's id is node_id_at(_, 0) == 1, NOT 0: a bare prefix would
        // make the root and the octant-0 child both 0.
        recurse(0, n_points_, 0, node_id_at(0, 0), root_center_, root_half_);
    }

    std::vector<NodeRec>& nodes() { return nodes_; }

    void set_point_count(uint64_t n) { n_points_ = n; }

private:
    void recurse(uint64_t begin, uint64_t end, int level, uint64_t node_id,
                 const double center[3], double half) {
        const uint64_t count = end - begin;

        bool want_split = count > params_.max_points_per_node;
        if (!want_split && params_.target_resolution > 0.0 && count > params_.min_points_per_node) {
            // node_cell_step() is the achievable 16-bit step for this cell.
            if (node_cell_step(static_cast<float>(half)) > params_.target_resolution)
                want_split = true;
        }
        if (level >= kMaxTreeLevel) want_split = false;

        // Octant ranges. The array is Morton-sorted, so every octant's members
        // are CONTIGUOUS - one linear walk finds all eight ranges, no sort, no
        // hash, no per-point node allocation. This is the whole reason to sort
        // by Morton key before bucketing.
        uint64_t oct_begin[8] = {0};
        uint64_t oct_end[8] = {0};
        uint8_t mask = 0;
        int found = 0;
        if (want_split) {
            int prev = -1;
            for (uint64_t i = begin; i < end; ++i) {
                const uint32_t o = octant_at(keys_[i], level);
                if (static_cast<int>(o) != prev) {
                    if (prev >= 0) oct_end[found - 1] = i;
                    oct_begin[found] = i;
                    found++;
                    prev = static_cast<int>(o);
                    mask = static_cast<uint8_t>(mask | (1u << o));
                }
            }
            if (found > 0) oct_end[found - 1] = end;
        }

        NodeRec rec;
        rec.node_id = node_id;
        rec.begin = begin;
        rec.end = end;
        rec.center[0] = center[0];
        rec.center[1] = center[1];
        rec.center[2] = center[2];
        rec.half_size = half;
        rec.level = static_cast<uint8_t>(level);
        rec.children_mask = want_split ? mask : 0;
        rec.is_leaf = !want_split;
        nodes_.push_back(rec);

        if (!want_split) return;

        const double child_half = 0.5 * half;
        for (int k = 0; k < found; ++k) {
            const uint32_t o = octant_at(keys_[oct_begin[k]], level);
            double cc[3];
            for (int a = 0; a < 3; ++a) {
                const bool high = ((o >> a) & 1u) != 0;
                cc[a] = center[a] + (high ? child_half : -child_half);
            }
            // Child id = the top 3*(level+1) bits of the key, level-tagged.
            // Equivalent and cheaper: parent_id << 3 | octant.
            const uint64_t child_id = node_id_at(keys_[oct_begin[k]], level + 1);
            recurse(oct_begin[k], oct_end[k], level + 1, child_id, cc, child_half);
        }
    }

    const uint64_t* keys_;
    const LNTBuildParams& params_;
    std::vector<NodeRec> nodes_;
    double root_center_[3] = {0, 0, 0};
    double root_half_ = 0.0;
    uint64_t n_points_ = 0;
};

}  // namespace detail

// ==========================================================================
// Stage 4 - the writer
// ==========================================================================
LNT_BUILD_FN LNTBuildStats LNTBuilder::build(const RawPoint* pts, size_t n,
                                             const char* out_path,
                                             const LNTBuildParams& params) {
    using clock = std::chrono::steady_clock;
    const auto t_start = clock::now();
    LNTBuildStats st;

    auto fail = [&](const char* what) {
        st.ok = false;
        st.error = what;
        return st;
    };
    if (pts == nullptr || n == 0) return fail("no input points");
    if (out_path == nullptr || out_path[0] == '\0') return fail("no output path");
    // Point identity is carried in a uint32 index array. That is the real
    // scale limit: 4.29e9 points. Individual NODE chunks have their own,
    // much smaller 32-bit BYTE length limit, checked per node below - a huge
    // input is fine as long as no single node exceeds it.
    if (n > 0xFFFFFFFFull) return fail("input exceeds the 32-bit point index");

    st.input_points = n;
    st.threads = detail::resolve_threads(params.threads);
    const unsigned T = st.threads;

    // ---- Stage 1: boundary scan + cubic envelope -------------------------
    auto t0 = clock::now();
    ScanResult sr = LNTBoundaryScanner::scan_cube(pts, n, T);
    Bounds3d cube = sr.bounds;
    if (!params.cube_bounds) {
        // Deliberately-uncubic mode, for contrast in a test.
        cube = LNTBoundaryScanner::scan(pts, n, T);
    }
    st.bounds = cube;
    st.scan_s = std::chrono::duration<double>(clock::now() - t0).count();

    // ---- Stage 2: quantise + Morton key ---------------------------------
    t0 = clock::now();
    std::vector<uint64_t> keys(n);
    std::vector<uint32_t> ids(n);
    detail::compute_morton_keys(pts, cube, keys.data(), ids.data(), n, T);
    st.morton_s = std::chrono::duration<double>(clock::now() - t0).count();

    // ---- Stage 3: radix sort + octree ------------------------------------
    t0 = clock::now();
    detail::RadixSort::sort(keys.data(), ids.data(), n, T);

    // The sort is stable by construction, but assert the observable
    // consequence: equal keys must appear in ascending source-index order.
    {
        bool stable = true;
        for (size_t i = 1; i < n; ++i) {
            if (keys[i] == keys[i - 1] && ids[i] < ids[i - 1]) { stable = false; break; }
        }
        st.stable_sort = stable;
    }
    st.sort_s = std::chrono::duration<double>(clock::now() - t0).count();

    t0 = clock::now();
    detail::OctreeBuilder tree(keys.data(), cube, params);
    tree.set_point_count(n);
    tree.build();
    std::vector<detail::NodeRec>& nodes = tree.nodes();
    st.node_count = nodes.size();
    for (const auto& r : nodes) {
        if (r.is_leaf) { st.leaf_count++; st.leaf_points += (r.end - r.begin); }
        if (r.level > st.max_depth) st.max_depth = r.level;
    }
    st.tree_s = std::chrono::duration<double>(clock::now() - t0).count();

    // ---- Stage 4: write --------------------------------------------------
    t0 = clock::now();
    std::ofstream f(out_path, std::ios::binary | std::ios::trunc);
    if (!f) return fail("cannot open output file");

    // Header first, with index_offset still 0: it is patched after the trailer
    // exists, so a half-written file can never advertise a valid trailer.
    LNTHeader hdr;
    std::memcpy(hdr.magic, kMagic, 4);
    hdr.version = kFormatVersion;
    hdr.reserved = 0;
    hdr.index_offset = 0;
    for (int i = 0; i < 3; ++i) {
        hdr.bounds_min[i] = cube.min[i];
        hdr.bounds_max[i] = cube.max[i];
    }
    f.write(reinterpret_cast<const char*>(&hdr),
            static_cast<std::streamsize>(sizeof(hdr)));

    std::vector<LNTPoint> buf;
    std::vector<LNTNodeEntry> entries;
    entries.reserve(nodes.size());

    for (detail::NodeRec& r : nodes) {
        const uint64_t count = r.end - r.begin;

        // Leaf: every point. Interior: a strided subsample, so a parent node
        // still has geometry to show before its children are read.
        uint64_t stride = 1;
        if (!r.is_leaf && count > params.parent_lod_target && params.parent_lod_target > 0) {
            stride = (count + params.parent_lod_target - 1) / params.parent_lod_target;
        }
        const uint64_t written = (count + stride - 1) / stride;
        if (written == 0) continue;
        if (written > kMaxChunkPoints) return fail("node chunk exceeds the 32-bit byte-length limit");

        // float32 offsets from the cube origin, NOT absolute world values:
        // an absolute float32 centre at 6.8e5 m is rounded to a 0.0625 m ulp,
        // which is coarser than the 9.16e-3 m cell step it is supposed to
        // anchor. Offsets stay under the cube side, where the ulp is 1.22e-4 m.
        const float cf[3] = {static_cast<float>(r.center[0] - cube.min[0]),
                             static_cast<float>(r.center[1] - cube.min[1]),
                             static_cast<float>(r.center[2] - cube.min[2])};
        const float hf = static_cast<float>(r.half_size);
        const double step = node_cell_step(hf);
        if (step > st.max_cell_step) st.max_cell_step = step;
        // Leaf-only precision, tracked separately from the all-nodes figure.
        // The ROOT always has a coarse cell (it spans the whole cube), so the
        // all-nodes maximum cannot show whether a deeper tree bought anything.
        // Leaf nodes are where full-detail geometry lives, so they are the
        // honest measure of a build's achieved precision.
        if (r.is_leaf && step > st.max_leaf_cell_step) st.max_leaf_cell_step = step;

        buf.clear();
        // Reserve before pushing: a reallocation inside the loop would still be
        // correct but would copy the whole chunk.
        buf.reserve(static_cast<size_t>(written));
        for (uint64_t i = r.begin; i < r.end; i += stride) {
            const RawPoint& p = pts[ids[static_cast<size_t>(i)]];
            LNTPoint q;
            const double w[3] = {p.x, p.y, p.z};
            uint16_t* rel[3] = {&q.x_rel, &q.y_rel, &q.z_rel};
            for (int a = 0; a < 3; ++a) {
                const double lo = cube.min[a] + double(cf[a]) - double(hf);
                const double hi = cube.min[a] + double(cf[a]) + double(hf);
                const uint32_t rv = quantize_to_16bit(w[a], lo, hi);
                *rel[a] = static_cast<uint16_t>(rv);
                // Measured decode error for this point, against the float32
                // anchor and the cube origin the reader will actually use.
                const double back = node_local_to_world(rv, cf[a], hf, cube.min[a]);
                const double e = std::fabs(back - w[a]);
                if (e > st.max_recon_error) st.max_recon_error = e;
                if (r.is_leaf && e > st.max_leaf_recon_error)
                    st.max_leaf_recon_error = e;
            }
            q.r = p.r;
            q.g = p.g;
            q.b = p.b;
            q.intensity = p.intensity;
            q.classification = p.classification;
            buf.push_back(q);
        }

        r.byte_offset = static_cast<uint64_t>(static_cast<std::streamoff>(f.tellp()));
        r.point_count = buf.size();
        r.byte_length = buf.size() * sizeof(LNTPoint);
        f.write(reinterpret_cast<const char*>(buf.data()),
                static_cast<std::streamsize>(r.byte_length));

        LNTNodeEntry e;
        std::memset(&e, 0, sizeof(e));
        e.node_id = r.node_id;
        e.byte_offset = r.byte_offset;
        e.byte_length = static_cast<uint32_t>(r.byte_length);
        e.point_count = static_cast<uint32_t>(r.point_count);
        for (int a = 0; a < 3; ++a) e.center[a] = cf[a];
        e.half_size = hf;
        e.children_mask = r.children_mask;
        e.tree_level = r.level;
        entries.push_back(e);

        if (!r.is_leaf) st.lod_points += r.point_count;
        st.written_points += r.point_count;
    }

    // ---- trailer ---------------------------------------------------------
    st.index_offset = static_cast<uint64_t>(static_cast<std::streamoff>(f.tellp()));
    LNTDirectoryHeader dh;
    dh.total_points = st.leaf_points;         // dataset size, LOD copies excluded
    dh.total_nodes = entries.size();
    f.write(reinterpret_cast<const char*>(&dh),
            static_cast<std::streamsize>(sizeof(dh)));
    if (!entries.empty()) {
        f.write(reinterpret_cast<const char*>(entries.data()),
                static_cast<std::streamsize>(entries.size() * sizeof(LNTNodeEntry)));
    }

    // Patch index_offset now that the trailer is on disk.
    hdr.index_offset = st.index_offset;
    f.seekp(0, std::ios::beg);
    f.write(reinterpret_cast<const char*>(&hdr),
            static_cast<std::streamsize>(sizeof(hdr)));
    f.flush();
    if (!f) return fail("write failed");
    f.close();

    st.file_bytes = st.index_offset + sizeof(LNTDirectoryHeader)
                    + entries.size() * sizeof(LNTNodeEntry);
    st.write_s = std::chrono::duration<double>(clock::now() - t0).count();
    st.node_count = entries.size();
    st.total_s = std::chrono::duration<double>(clock::now() - t_start).count();
    st.ok = st.written_points > 0;
    if (!st.ok) st.error = "nothing written";
    return st;
}

}  // namespace lnt

#endif  // LNT_BUILDER_IMPL_H
