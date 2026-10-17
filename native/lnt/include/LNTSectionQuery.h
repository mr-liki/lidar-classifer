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

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "LNTFormat.h"
#include "LNTMath.h"
#include "LNTReader.h"

namespace lnt {

// UTF-8 .lnt path -> wide, for CreateFileW (tried as UTF-8 first, then the ANSI code
// page, like surface_archive::widen in LNTSurfaceArchive.h).
inline std::wstring section_path_wide(const std::string& path) {
    for (UINT cp : {UINT(CP_UTF8), UINT(CP_ACP)}) {
        const DWORD flags = cp == CP_UTF8 ? MB_ERR_INVALID_CHARS : 0;
        const int n = MultiByteToWideChar(cp, flags, path.data(), int(path.size()), nullptr, 0);
        if (n <= 0) continue;
        std::wstring w(size_t(n), L'\0');
        MultiByteToWideChar(cp, flags, path.data(), int(path.size()), &w[0], n);
        return w;
    }
    return std::wstring(path.begin(), path.end());
}

struct SectionHit {
    uint32_t node;     // index into Reader::nodes()
    uint32_t local;    // index inside that leaf's chunk
};

// ---- directory geometry needed to find LOD copies ----------------------------
struct SectionEditIndex {
    std::vector<uint64_t> leaf_prefix;   // leaf points stored BEFORE directory entry i
    std::vector<uint64_t> subtree_pts;   // leaf points under entry i (own count if a leaf)
    std::unordered_map<uint64_t, uint32_t> dir_index;   // node_id -> directory index
    std::unordered_map<uint32_t, uint64_t> lod_stride;  // ancestor -> last verified copy stride
    std::unordered_set<uint32_t> lod_stride_miss;       // ancestor -> no stride verified (static)

    // One Win32 write handle on the .lnt, kept open across edits: opening a large
    // file for writing costs hundreds of ms on Windows (security scanning) on every
    // brush stroke. Same access/sharing as surface_archive::File::open_write
    // (GENERIC_READ|GENERIC_WRITE, FILE_SHARE_READ|FILE_SHARE_WRITE, OPEN_EXISTING):
    // positional WriteFile never disturbs the reader, and the surface archive only
    // appends/truncates past base_end, clear of every point chunk below.
    HANDLE write_handle = INVALID_HANDLE_VALUE;
    std::string file_path;
    ~SectionEditIndex() { close_file(); }
    SectionEditIndex() = default;
    SectionEditIndex(const SectionEditIndex&) = delete;
    SectionEditIndex& operator=(const SectionEditIndex&) = delete;
    SectionEditIndex(SectionEditIndex&&) = delete;
    SectionEditIndex& operator=(SectionEditIndex&&) = delete;
    HANDLE writer(const std::string& path) {
        if (write_handle != INVALID_HANDLE_VALUE && file_path == path) return write_handle;
        close_file();
        write_handle = CreateFileW(section_path_wide(path).c_str(), GENERIC_READ | GENERIC_WRITE,
                                   FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                   OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (write_handle == INVALID_HANDLE_VALUE) return INVALID_HANDLE_VALUE;
        file_path = path;
        return write_handle;
    }
    void close_file() {
        if (write_handle != INVALID_HANDLE_VALUE) CloseHandle(write_handle);
        write_handle = INVALID_HANDLE_VALUE;
        file_path.clear();
    }

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

// Every leaf point a plan-view brush stroke covers: the brush (round: radius-r disc;
// square: axis-aligned square of half side r) swept along each segment of the XY path
// (n points, x0 y0 x1 y1 ...). Same formulas as brush_sweep() in
// gui/native_section_classify.py, so the two select identical records. Only the swept
// area is returned (a stroke's bounding box can hold many times more points).
inline int64_t section_query_stroke(Reader& rd, const double* path, uint64_t n, double r, bool square,
                                    std::vector<SectionHit>& out, std::string& err,
                                    uint64_t max_points = 0) {
    out.clear();
    err.clear();
    if (!rd.ok()) { err = "reader is not open"; return -1; }
    if (!path || n == 0 || !(r > 0.0)) return 0;
    struct Seg { double ax, ay, dx, dy, len2, lo[2], hi[2]; };
    std::vector<Seg> segs;
    const uint64_t ns_count = n == 1 ? 1 : n - 1;
    segs.reserve(ns_count);
    double slo[2] = {1e300, 1e300}, shi[2] = {-1e300, -1e300};
    for (uint64_t i = 0; i < ns_count; ++i) {
        const double ax = path[i * 2], ay = path[i * 2 + 1];
        const double bx = n == 1 ? ax : path[i * 2 + 2], by = n == 1 ? ay : path[i * 2 + 3];
        Seg s{ax, ay, bx - ax, by - ay, 0.0, {std::min(ax, bx) - r, std::min(ay, by) - r},
              {std::max(ax, bx) + r, std::max(ay, by) + r}};
        s.len2 = s.dx * s.dx + s.dy * s.dy;
        for (int a = 0; a < 2; ++a) { slo[a] = std::min(slo[a], s.lo[a]); shi[a] = std::max(shi[a], s.hi[a]); }
        segs.push_back(s);
    }
    auto inside = [&](const Seg& s, double x, double y) {
        const double px = x - s.ax, pz = y - s.ay;
        if (square) {
            double t_lo = 0.0, t_hi = 1.0;
            const double p2[2] = {px, pz}, d2[2] = {s.dx, s.dy};
            for (int a = 0; a < 2; ++a) {
                if (d2[a] == 0.0) { if (std::fabs(p2[a]) > r) return false; continue; }
                const double t1 = (p2[a] - r) / d2[a], t2 = (p2[a] + r) / d2[a];
                t_lo = std::max(t_lo, std::min(t1, t2));
                t_hi = std::min(t_hi, std::max(t1, t2));
            }
            return t_lo <= t_hi;
        }
        double t = 0.0;
        if (s.len2 > 0.0) t = std::min(1.0, std::max(0.0, (px * s.dx + pz * s.dy) / s.len2));
        const double ex = px - t * s.dx, ez = pz - t * s.dy;
        return ex * ex + ez * ez <= r * r;
    };
    const std::vector<LNTNodeEntry>& ns = rd.nodes();
    const double* org = rd.header().bounds_min;
    std::vector<LNTPoint> raw;
    std::vector<const Seg*> close_segs;
    for (std::size_t i = 0; i < ns.size(); ++i) {
        const LNTNodeEntry& e = ns[i];
        if (!is_leaf_node(e) || e.point_count == 0) continue;
        const BoundingBox cell = rd.bounds_of(e);
        if (cell.max[0] < slo[0] || cell.min[0] > shi[0] || cell.max[1] < slo[1] || cell.min[1] > shi[1]) continue;
        close_segs.clear();
        for (const Seg& s : segs)
            if (!(cell.max[0] < s.lo[0] || cell.min[0] > s.hi[0] || cell.max[1] < s.lo[1] || cell.min[1] > s.hi[1]))
                close_segs.push_back(&s);
        if (close_segs.empty()) continue;
        // A cell whose four corners lie in one segment's (convex) swept shape is covered whole.
        bool whole = false;
        for (const Seg* s : close_segs)
            if (inside(*s, cell.min[0], cell.min[1]) && inside(*s, cell.max[0], cell.min[1]) &&
                inside(*s, cell.min[0], cell.max[1]) && inside(*s, cell.max[0], cell.max[1])) { whole = true; break; }
        if (!rd.read_node_raw(e, raw)) { err = rd.last_error(); out.clear(); return -1; }
        for (uint32_t k = 0; k < raw.size(); ++k) {
            if (!whole) {
                const RawPoint p = decode_point(raw[k], e, org);
                bool hit = false;
                for (const Seg* s : close_segs) if (inside(*s, p.x, p.y)) { hit = true; break; }
                if (!hit) continue;
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
// One record whose class an edit changed: the chunk it lives in (leaf or LOD copy), its
// index there, its exact decoded position and the new class. The viewer recolours the
// surface vertices made from it without re-meshing.
struct SectionRecordEdit {
    uint32_t node = 0, local = 0;
    double x = 0.0, y = 0.0, z = 0.0;
    uint8_t cls = 0, old_cls = 0;
};

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
                                SectionEditStats* stats = nullptr,
                                std::vector<SectionRecordEdit>* records = nullptr,
                                bool preview_only = false, const uint8_t* preview_old = nullptr) {
    touched.clear();
    if (records) records->clear();
    err.clear();
    if (n == 0) return true;
    if (!rd.ok()) { err = "reader is not open"; return false; }
    if (!idx.built()) idx.build(rd);
    SectionEditStats st;
    const std::vector<LNTNodeEntry>& ns = rd.nodes();
    const double* org = rd.header().bounds_min;

    // Per leaf: (local, new class, shown class) - the shown class is preview_old when
    // given (classes on screen can differ from the file before a deferred write).
    struct LeafEdit { uint32_t local; uint8_t cls, old; };
    std::unordered_map<uint32_t, std::vector<LeafEdit>> by_leaf;
    for (uint64_t j = 0; j < n; ++j) {
        if (node_idx[j] >= ns.size() || !is_leaf_node(ns[node_idx[j]]) ||
            local_idx[j] >= ns[node_idx[j]].point_count) {
            err = "point identity out of range";
            return false;
        }
        by_leaf[node_idx[j]].push_back({local_idx[j], new_class[j], preview_old ? preview_old[j] : uint8_t(0)});
    }

    HANDLE wf = INVALID_HANDLE_VALUE;
    if (!preview_only) {
        wf = idx.writer(rd.path());
        if (wf == INVALID_HANDLE_VALUE) {
            err = "cannot open '" + rd.path() + "' for writing";
            return false;
        }
    }
    // Writes records [lo, hi] of a chunk: only the run an edit changed, not the chunk.
    // Positional WriteFile needs no flush; the preview path never opens the file.
    auto write_chunk = [&](const LNTNodeEntry& e, const std::vector<LNTPoint>& chunk, uint64_t lo, uint64_t hi) {
        if (preview_only) return true;
        std::unique_lock<std::shared_mutex> guard;
        if (io) guard = std::unique_lock<std::shared_mutex>(*io);
        uint64_t off = e.byte_offset + lo * sizeof(LNTPoint);
        const char* src = reinterpret_cast<const char*>(chunk.data() + lo);
        uint64_t left = (hi - lo + 1) * sizeof(LNTPoint);
        while (left) {
            OVERLAPPED ov{};
            ov.Offset = DWORD(off);
            ov.OffsetHigh = DWORD(off >> 32);
            const DWORD want = DWORD(left > (8u << 20) ? (8u << 20) : left);
            DWORD put = 0;
            if (!WriteFile(wf, src, want, &put, &ov) || put != want) {
                idx.close_file();                               // reopened by the next edit
                return false;
            }
            src += put; off += put; left -= put;
        }
        return true;
    };

    std::unordered_map<uint32_t, std::vector<LNTPoint>> leaf_raw;     // edited leaves (post-edit)
    std::unordered_map<uint32_t, std::vector<uint8_t>> leaf_was;      // their shown classes
    std::vector<LNTPoint> raw;

    for (auto& kv : by_leaf) {
        const uint32_t li = kv.first;
        const LNTNodeEntry& le = ns[li];
        if (!rd.read_node_raw(le, raw)) { err = rd.last_error(); return false; }
        bool changed = false;
        uint64_t lo = UINT64_MAX, hi = 0;
        std::vector<uint8_t> was_cls(raw.size());
        for (std::size_t k = 0; k < raw.size(); ++k) was_cls[k] = raw[k].classification;
        for (auto& ed : kv.second) {
            if (preview_old) was_cls[ed.local] = ed.old;
            if (preview_only || raw[ed.local].classification != ed.cls) {
                const uint8_t was = was_cls[ed.local];
                raw[ed.local].classification = ed.cls; changed = true;
                lo = std::min<uint64_t>(lo, ed.local); hi = std::max<uint64_t>(hi, ed.local);
                if (records) {
                    const RawPoint q = decode_point(raw[ed.local], le, org);
                    records->push_back({li, ed.local, q.x, q.y, q.z, ed.cls, was});
                }
            }
        }
        if (!changed) continue;
        leaf_was[li] = std::move(was_cls);
        if (!write_chunk(le, raw, lo, hi)) { err = "write failed"; return false; }
        ++st.leaves_rewritten;
        touched.push_back(li);
        leaf_raw[li] = std::move(raw);
    }

    // ---- LOD copies --------------------------------------------------------------
    // Ancestors are grouped by LEAF (hundreds), not by point (millions): a stroke's
    // edited points fan out to ~depth ancestors each, and materializing one record per
    // (point, ancestor) pair costs hundreds of MB of transient vectors. A stride is
    // verified by walking the copies it CLAIMS (sorted positions begin + k*stride)
    // inside the edited leaves - exact for wide strides (few copies) and capped at a
    // handful of tests for narrow ones; every copy is still checked individually when
    // it is applied.
    std::unordered_map<uint32_t, std::vector<uint32_t>> anc_leaves;   // anc -> edited leaves under it
    anc_leaves.reserve(leaf_raw.size() * 2 + 1);
    std::unordered_map<uint32_t, std::vector<char>> edited_mark;      // leaf -> edited locals
    edited_mark.reserve(leaf_raw.size() * 2 + 1);
    for (auto& kv : leaf_raw) {
        const uint32_t li = kv.first;
        for (uint64_t pid = ns[li].node_id >> 3; pid != 0; pid >>= 3) {
            auto it = idx.dir_index.find(pid);
            if (it != idx.dir_index.end()) anc_leaves[it->second].push_back(li);
        }
        std::vector<char>& mark = edited_mark[li];
        mark.assign(ns[li].point_count, 0);
        for (auto& ed : by_leaf[li]) mark[ed.local] = 1;
    }
    for (auto& pa : anc_leaves) {
        const uint32_t ai = pa.first;
        const std::vector<uint32_t>& leaves = pa.second;
        const LNTNodeEntry& ae = ns[ai];
        const uint64_t c = idx.subtree_pts[ai], w = ae.point_count;
        if (w == 0 || c == 0) continue;
        if (!rd.read_node_raw(ae, raw)) { err = rd.last_error(); return false; }
        const double step = node_cell_step(ae.half_size);
        const double tol = 1.5 * step + 1e-6;

        // candidate strides that reproduce this ancestor's point count
        std::vector<uint64_t> cands;
        auto known = idx.lod_stride.find(ai);
        if (known != idx.lod_stride.end()) cands.push_back(known->second);   // verified before
        else if (idx.lod_stride_miss.find(ai) != idx.lod_stride_miss.end()) {
            ++st.lod_ancestors_skipped; continue;                            // verified absent before
        }
        if (c <= w) cands.push_back(1);
        else {
            const uint64_t stride_lo = (c + w - 1) / w;
            const uint64_t stride_hi = (w > 1) ? (c - 1) / (w - 1) : c;
            for (uint64_t s = stride_lo; s <= stride_hi && cands.size() < 64; ++s)
                if ((c + s - 1) / s == w) cands.push_back(s);
        }
        auto matches = [&](uint64_t s, uint64_t base, uint32_t li, uint32_t local,
                           const LNTPoint& b, uint64_t& k_out) {
            const uint64_t rel = base + local;
            if (rel % s != 0) return 0;                       // not a copy under this stride
            const uint64_t k = rel / s;
            if (k >= w) return -1;
            const LNTPoint& a = raw[k];
            const RawPoint pa_ = decode_point(a, ae, org), pb_ = decode_point(b, ns[li], org);
            if (std::fabs(pa_.x - pb_.x) > tol || std::fabs(pa_.y - pb_.y) > tol ||
                std::fabs(pa_.z - pb_.z) > tol || a.r != b.r || a.g != b.g || a.b != b.b ||
                a.intensity != b.intensity)
                return -1;                                    // the entry is some other point
            k_out = k;
            return 1;
        };
        // The copies candidate s claims inside the edited leaves decide it: at most
        // 64 edited copies are compared (a wrong stride is caught by the first few),
        // so wide strides cost almost nothing and narrow ones stay bounded. Every copy
        // is still checked individually when it is applied below.
        auto trial = [&](uint64_t s, int& hit) {
            hit = 0;
            int tested = 0;
            const uint64_t begin = idx.leaf_prefix[ai];
            for (uint32_t li : leaves) {
                const uint64_t P = idx.leaf_prefix[li];
                const uint64_t cnt = ns[li].point_count;
                uint64_t k_lo = (P <= begin) ? 0 : (P - begin + s - 1) / s;
                uint64_t k_hi = (P + cnt <= begin) ? 0 : (P + cnt - 1 - begin) / s + 1;
                if (k_hi > w) k_hi = w;
                const std::vector<char>& mark = edited_mark[li];
                const std::vector<LNTPoint>& lraw = leaf_raw[li];
                for (uint64_t kk = k_lo; kk < k_hi && tested < 64; ++kk) {
                    const uint32_t local = uint32_t(begin + kk * s - P);
                    if (!mark[local]) continue;
                    const LNTPoint& a = raw[kk];
                    const LNTPoint& b = lraw[local];
                    const RawPoint pa_ = decode_point(a, ae, org), pb_ = decode_point(b, ns[li], org);
                    if (std::fabs(pa_.x - pb_.x) > tol || std::fabs(pa_.y - pb_.y) > tol ||
                        std::fabs(pa_.z - pb_.z) > tol || a.r != b.r || a.g != b.g || a.b != b.b ||
                        a.intensity != b.intensity)
                        return false;                             // the entry is some other point
                    ++hit; ++tested;
                }
                if (tested >= 64) break;
            }
            return true;
        };
        uint64_t best = 0;
        {
            int hit = 0;
            for (uint64_t s : cands)
                if (trial(s, hit) && hit > 0) { best = s; break; }
        }
        if (best == 0) {
            ++st.lod_ancestors_skipped;
            idx.lod_stride_miss.insert(ai);   // static property of the file: never re-search
            continue;
        }
        idx.lod_stride[ai] = best;
        bool changed = false;
        uint64_t lo = UINT64_MAX, hi = 0;
        for (uint32_t li : leaves) {
            const uint64_t base = idx.leaf_prefix[li] - idx.leaf_prefix[ai];
            const std::vector<LNTPoint>& lraw = leaf_raw[li];
            const std::vector<uint8_t>& lwas = leaf_was[li];
            for (auto& ed : by_leaf[li]) {
                uint64_t k;
                if (matches(best, base, li, ed.local, lraw[ed.local], k) <= 0) continue;
                const uint8_t want = lraw[ed.local].classification;
                if (preview_only || raw[k].classification != want) {
                    const uint8_t was = preview_old ? lwas[ed.local] : raw[k].classification;
                    raw[k].classification = want; changed = true; ++st.lod_copies_updated;
                    lo = std::min(lo, k); hi = std::max(hi, k);
                    if (records) {
                        const RawPoint q = decode_point(raw[k], ae, org);
                        records->push_back({ai, uint32_t(k), q.x, q.y, q.z, want, was});
                    }
                }
            }
        }
        if (!changed) continue;
        if (!write_chunk(ae, raw, lo, hi)) { err = "write failed"; return false; }
        ++st.lod_chunks_rewritten;
        touched.push_back(ai);
    }
    if (stats) *stats = st;
    return true;
}

}  // namespace lnt

#endif  // LNT_SECTION_QUERY_H
