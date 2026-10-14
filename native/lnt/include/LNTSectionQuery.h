// LNTSectionQuery.h - point query and classification write-back on a .lnt file.
//
// What the Cross Section / Cut Section tools need from a streamed cloud that is never
// held in RAM as a whole:
//
//   1. QUERY   every point inside a world box (the section corridor), each tagged with
//              a stable identity (directory node index, index inside that leaf chunk).
//   2. EDIT    set the classification of points by identity, IN THE FILE, so a section
//              edit survives closing the file and shows in the main view.
//
// ONLY LEAF CHUNKS ARE QUERIED. A leaf holds every point of its cell exactly once; an
// interior node holds a strided LOD copy of descendants' points, which would return a
// point more than once (see LNTBuilderImpl.h, stride = ceil(count / parent_lod_target)).
//
// KEEPING THE LOD COPIES IN STEP. An edited leaf point also exists, for display, as a
// copy in some ancestors' chunks. Leaving those stale would show the OLD class at
// coarse zoom levels. The copy is found by arithmetic, not by searching: the builder
// writes interior chunk entry k from sorted position  begin + k * stride  of the
// Morton-sorted points, and a leaf's points ARE a contiguous run of that order. The
// directory is pre-order, so
//     sorted position of (leaf, i)   = (leaf points before the leaf in the directory) + i
//     an ancestor's range            = [leaf points before it, + points under it)
// and the point is a copy in that ancestor iff (position - begin) % stride == 0, at
// entry (position - begin) / stride. The stride itself is not stored; it is the one
// that reproduces the ancestor's point count AND whose entry matches the edited point
// (coordinates within the ancestor cell's quantisation step, same colour and intensity).
// A candidate stride that fails that check is never applied: the LOD copy is left alone
// rather than corrupting a different point.

#ifndef LNT_SECTION_QUERY_H
#define LNT_SECTION_QUERY_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "LNTFormat.h"
#include "LNTMath.h"
#include "LNTReader.h"

namespace lnt {

struct SectionHit {
    uint32_t node;     // index into Reader::nodes()
    uint32_t local;    // index inside that leaf's chunk
};

// ---- directory geometry needed to find LOD copies ----------------------------
struct SectionEditIndex {
    std::vector<uint64_t> leaf_prefix;   // leaf points stored BEFORE directory entry i
    std::vector<uint64_t> subtree_pts;   // leaf points under entry i (own count if a leaf)
    std::unordered_map<uint64_t, uint32_t> dir_index;   // node_id -> directory index

    bool built() const { return !leaf_prefix.empty(); }
    void build(const Reader& rd) {
        const std::vector<LNTNodeEntry>& ns = rd.nodes();
        const std::size_t n = ns.size();
        leaf_prefix.assign(n, 0);
        subtree_pts.assign(n, 0);
        dir_index.clear();
        dir_index.reserve(n * 2);
        uint64_t run = 0;
        for (std::size_t i = 0; i < n; ++i) {
            dir_index[ns[i].node_id] = uint32_t(i);
            leaf_prefix[i] = run;
            if (is_leaf_node(ns[i])) { run += ns[i].point_count; subtree_pts[i] = ns[i].point_count; }
        }
        // Pre-order table: every child follows its parent, so a reverse walk finishes a
        // child's subtree before it is added to the parent's.
        for (std::size_t i = n; i-- > 1;) {
            auto it = dir_index.find(ns[i].node_id >> 3);      // child_id >> 3 == parent_id
            if (it != dir_index.end()) subtree_pts[it->second] += subtree_pts[i];
        }
    }
};

// ---- QUERY ---------------------------------------------------------------------
// Every leaf point inside the CLOSED box [mn, mx]. Returns the count, or -1 on error.
// `max_points` (0 = unlimited) stops a query that would not fit in memory: it returns
// -2 and leaves `out` empty, so the caller can ask for a smaller region instead.
inline int64_t section_query_aabb(Reader& rd, const double mn[3], const double mx[3],
                                  std::vector<SectionHit>& out, std::string& err,
                                  uint64_t max_points = 0) {
    out.clear();
    err.clear();
    if (!rd.ok()) { err = "reader is not open"; return -1; }
    const BoundingBox box = BoundingBox::from_min_max(mn[0], mn[1], mn[2], mx[0], mx[1], mx[2]);
    if (box.is_empty()) return 0;
    const std::vector<LNTNodeEntry>& ns = rd.nodes();
    const double* org = rd.header().bounds_min;
    std::vector<LNTPoint> raw;
    for (std::size_t i = 0; i < ns.size(); ++i) {
        const LNTNodeEntry& e = ns[i];
        if (!is_leaf_node(e) || e.point_count == 0) continue;
        if (!rd.bounds_of(e).intersects(box)) continue;
        if (!rd.read_node_raw(e, raw)) { err = rd.last_error(); out.clear(); return -1; }
        // A cell that lies wholly inside the box needs no per-point test.
        const BoundingBox cell = rd.bounds_of(e);
        bool whole = true;
        for (int a = 0; a < 3; ++a)
            if (cell.min[a] < box.min[a] || cell.max[a] > box.max[a]) { whole = false; break; }
        for (uint32_t k = 0; k < raw.size(); ++k) {
            if (!whole) {
                const RawPoint p = decode_point(raw[k], e, org);
                if (!box.contains(p)) continue;
            }
            out.push_back({uint32_t(i), k});
        }
        if (max_points && out.size() > max_points) { out.clear(); err = "too many points"; return -2; }
    }
    return int64_t(out.size());
}

// Copies the hits' attributes out (any output pointer may be null). xyz is n*3 float64,
// rgb n*3 uint8. Hits must be in the order section_query_aabb produced (grouped by node).
inline bool section_fetch(Reader& rd, const std::vector<SectionHit>& hits, double* xyz,
                          uint8_t* cls, uint8_t* inten, uint8_t* rgb, uint32_t* node_idx,
                          uint32_t* local_idx) {
    const std::vector<LNTNodeEntry>& ns = rd.nodes();
    const double* org = rd.header().bounds_min;
    std::vector<LNTPoint> raw;
    uint32_t cur = 0xFFFFFFFFu;
    for (std::size_t j = 0; j < hits.size(); ++j) {
        const SectionHit& h = hits[j];
        if (h.node >= ns.size()) return false;
        if (h.node != cur) {
            if (!rd.read_node_raw(ns[h.node], raw)) return false;
            cur = h.node;
        }
        if (h.local >= raw.size()) return false;
        const LNTPoint& q = raw[h.local];
        if (xyz) {
            const RawPoint p = decode_point(q, ns[h.node], org);
            xyz[j * 3] = p.x; xyz[j * 3 + 1] = p.y; xyz[j * 3 + 2] = p.z;
        }
        if (cls) cls[j] = q.classification;
        if (inten) inten[j] = q.intensity;
        if (rgb) { rgb[j * 3] = q.r; rgb[j * 3 + 1] = q.g; rgb[j * 3 + 2] = q.b; }
        if (node_idx) node_idx[j] = h.node;
        if (local_idx) local_idx[j] = h.local;
    }
    return true;
}

// ---- EDIT ----------------------------------------------------------------------
// Sets the classification of the given leaf points IN THE FILE, plus their LOD copies.
// `io` (optional) is taken EXCLUSIVELY around each chunk write, so a loader thread that
// holds it shared can never read half a chunk. `touched` receives the directory index of
// every chunk that was rewritten (leaves and ancestors) so the viewer can refresh them.
struct SectionEditStats {
    uint64_t leaves_rewritten = 0;
    uint64_t lod_chunks_rewritten = 0;
    uint64_t lod_copies_updated = 0;
    uint64_t lod_ancestors_skipped = 0;   // stride could not be verified: left untouched
};

inline bool section_set_classes(Reader& rd, SectionEditIndex& idx, std::shared_mutex* io,
                                const uint32_t* node_idx, const uint32_t* local_idx,
                                const uint8_t* new_class, uint64_t n,
                                std::vector<uint32_t>& touched, std::string& err,
                                SectionEditStats* stats = nullptr) {
    touched.clear();
    err.clear();
    if (n == 0) return true;
    if (!rd.ok()) { err = "reader is not open"; return false; }
    if (!idx.built()) idx.build(rd);
    SectionEditStats st;
    const std::vector<LNTNodeEntry>& ns = rd.nodes();
    const double* org = rd.header().bounds_min;

    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint8_t>>> by_leaf;
    for (uint64_t j = 0; j < n; ++j) {
        if (node_idx[j] >= ns.size() || !is_leaf_node(ns[node_idx[j]]) ||
            local_idx[j] >= ns[node_idx[j]].point_count) {
            err = "point identity out of range";
            return false;
        }
        by_leaf[node_idx[j]].emplace_back(local_idx[j], new_class[j]);
    }

    std::fstream wf(rd.path(), std::ios::in | std::ios::out | std::ios::binary);
    if (!wf.good()) { err = "cannot open '" + rd.path() + "' for writing"; return false; }
    auto write_chunk = [&](const LNTNodeEntry& e, const std::vector<LNTPoint>& chunk) {
        std::unique_lock<std::shared_mutex> guard;
        if (io) guard = std::unique_lock<std::shared_mutex>(*io);
        wf.clear();
        wf.seekp(std::streamoff(e.byte_offset), std::ios::beg);
        wf.write(reinterpret_cast<const char*>(chunk.data()),
                 std::streamsize(chunk.size() * sizeof(LNTPoint)));
        wf.flush();
        return wf.good();
    };

    std::unordered_map<uint32_t, std::vector<LNTPoint>> leaf_raw;     // edited leaves (post-edit)
    std::vector<LNTPoint> raw;

    for (auto& kv : by_leaf) {
        const uint32_t li = kv.first;
        const LNTNodeEntry& le = ns[li];
        if (!rd.read_node_raw(le, raw)) { err = rd.last_error(); return false; }
        bool changed = false;
        for (auto& ed : kv.second)
            if (raw[ed.first].classification != ed.second) { raw[ed.first].classification = ed.second; changed = true; }
        if (!changed) continue;
        if (!write_chunk(le, raw)) { err = "write failed"; return false; }
        ++st.leaves_rewritten;
        touched.push_back(li);
        leaf_raw[li] = raw;
    }

    // ---- LOD copies --------------------------------------------------------------
    // group edited points by ancestor
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> per_ancestor;   // anc -> (leaf, local)
    for (auto& kv : leaf_raw) {
        const uint32_t li = kv.first;
        for (auto& ed : by_leaf[li])
            for (uint64_t pid = ns[li].node_id >> 3; pid != 0; pid >>= 3) {
                auto it = idx.dir_index.find(pid);
                if (it != idx.dir_index.end()) per_ancestor[it->second].emplace_back(li, ed.first);
            }
    }
    for (auto& pa : per_ancestor) {
        const uint32_t ai = pa.first;
        const LNTNodeEntry& ae = ns[ai];
        const uint64_t c = idx.subtree_pts[ai], w = ae.point_count;
        if (w == 0 || c == 0) continue;
        if (!rd.read_node_raw(ae, raw)) { err = rd.last_error(); return false; }
        const double step = node_cell_step(ae.half_size);

        // candidate strides that reproduce this ancestor's point count
        std::vector<uint64_t> cands;
        if (c <= w) cands.push_back(1);
        else {
            const uint64_t lo = (c + w - 1) / w;
            const uint64_t hi = (w > 1) ? (c - 1) / (w - 1) : c;
            for (uint64_t s = lo; s <= hi && cands.size() < 64; ++s)
                if ((c + s - 1) / s == w) cands.push_back(s);
        }
        auto matches = [&](uint64_t s, uint32_t li, uint32_t local, uint64_t& k_out) {
            const uint64_t rel = (idx.leaf_prefix[li] - idx.leaf_prefix[ai]) + local;
            if (rel % s != 0) return 0;                       // not a copy under this stride
            const uint64_t k = rel / s;
            if (k >= w) return -1;
            const LNTPoint& a = raw[k];
            const LNTPoint& b = leaf_raw[li][local];
            const RawPoint pa_ = decode_point(a, ae, org), pb_ = decode_point(b, ns[li], org);
            const double tol = 1.5 * step + 1e-6;
            if (std::fabs(pa_.x - pb_.x) > tol || std::fabs(pa_.y - pb_.y) > tol ||
                std::fabs(pa_.z - pb_.z) > tol || a.r != b.r || a.g != b.g || a.b != b.b ||
                a.intensity != b.intensity)
                return -1;                                    // the entry is some other point
            k_out = k;
            return 1;
        };
        uint64_t best = 0;
        for (uint64_t s : cands) {
            int hit = 0, bad = 0;
            uint64_t k;
            for (auto& pt : pa.second) {
                const int m = matches(s, pt.first, pt.second, k);
                if (m > 0) ++hit; else if (m < 0) ++bad;
            }
            if (bad == 0 && hit > 0) { best = s; break; }
        }
        if (best == 0) { ++st.lod_ancestors_skipped; continue; }
        bool changed = false;
        for (auto& pt : pa.second) {
            uint64_t k;
            if (matches(best, pt.first, pt.second, k) <= 0) continue;
            const uint8_t want = leaf_raw[pt.first][pt.second].classification;
            if (raw[k].classification != want) { raw[k].classification = want; changed = true; ++st.lod_copies_updated; }
        }
        if (!changed) continue;
        if (!write_chunk(ae, raw)) { err = "write failed"; return false; }
        ++st.lod_chunks_rewritten;
        touched.push_back(ai);
    }
    wf.close();
    if (stats) *stats = st;
    return true;
}

}  // namespace lnt

#endif  // LNT_SECTION_QUERY_H
