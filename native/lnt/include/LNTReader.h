// LNTReader.h - read side of the .lnt single-file format.
//
// Opens a .lnt once, validates the 64-byte header and the trailer, loads the
// whole node directory into memory, and then streams point payloads on demand.
// Nothing but the directory is resident: a node's chunk is read from disk only
// when it is asked for, so opening a billion-point file costs one directory
// read and no payload traffic at all.
//
// WHY THE DIRECTORY IS THE INTERESTING PART. The node table is sorted by Morton
// key and each LNTNodeEntry carries its own cell (centre + half_size), so node
// bounding boxes are available without touching a single point. Culling
// therefore happens at node granularity: `query_nodes_in_bounds` is a scan of
// the directory, not of the dataset. That is what makes the render path able to
// select what to draw with no payload reads - see LNTRenderBridge.h, where a
// node range in the directory becomes a draw range in the GPU arena.
//
// Two INVERSE functions exist in this codebase and they must not be confused:
//   node_local_to_world()   (LNTMath.h)  the per-point dequantiser
//   node_cell_step()        (LNTMath.h)  the width of one 16-bit payload level
// This reader uses the former for coordinates and the latter as the honest
// error bound, because they answer different questions. See decode_point().
//
// HEADER-ONLY DEFINITIONS, deliberately. The .lnt pipeline has two supported
// builds - a single command (`g++ -O3 -std=c++17 main.cpp -o main`, which has
// no second translation unit to link against) and a split form
// (`g++ -O3 -std=c++17 main.cpp LNTReader.cpp -o main`). Defining the class
// only in LNTReader.cpp would break the first; defining it inline keeps both
// working from one source of truth, exactly as LNTBuilderImpl.h does for the
// builder. LNTReader.cpp remains the split build's entry point.

#ifndef LNT_READER_H
#define LNT_READER_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "LNTFormat.h"
#include "LNTMath.h"
#include "LNTTypes.h"

namespace lnt {

// ========================================================================
// BoundingBox - an axis-aligned world-space box
// ========================================================================
// Distinct from Bounds3d (LNTTypes.h) on purpose. Bounds3d is a BUILDER
// accumulator: empty-initialised, expand-only, and it owns the cube-snapping
// rule that the writer applies. A query box is none of those things - it is a
// fixed region to test against - so sharing the type would mean sharing a
// set_empty() convention that only makes sense on one side.
struct BoundingBox {
    double min[3];
    double max[3];

    // A default-constructed box is the degenerate box at the origin, which
    // matches only the exact origin point. Use from_min_max / from_center_half
    // before relying on it.
    BoundingBox() {
        for (int i = 0; i < 3; ++i) { min[i] = 0.0; max[i] = 0.0; }
    }

    static BoundingBox from_min_max(double x0, double y0, double z0,
                                    double x1, double y1, double z1) {
        BoundingBox b;
        b.min[0] = x0; b.min[1] = y0; b.min[2] = z0;
        b.max[0] = x1; b.max[1] = y1; b.max[2] = z1;
        return b;
    }

    static BoundingBox from_center_half(const double c[3], double half) {
        BoundingBox b;
        for (int i = 0; i < 3; ++i) {
            b.min[i] = c[i] - half;
            b.max[i] = c[i] + half;
        }
        return b;
    }

    static BoundingBox from_bounds(const Bounds3d& s) {
        BoundingBox b;
        for (int i = 0; i < 3; ++i) { b.min[i] = s.min[i]; b.max[i] = s.max[i]; }
        return b;
    }

    void set_empty() {
        for (int i = 0; i < 3; ++i) { min[i] = +HUGE_VAL; max[i] = -HUGE_VAL; }
    }

    bool is_empty() const {
        return !(max[0] >= min[0] && max[1] >= min[1] && max[2] >= min[2]);
    }

    double extent(int axis) const { return max[axis] - min[axis]; }
    double mid(int axis) const { return 0.5 * (min[axis] + max[axis]); }
    bool is_cube(double tol = 1e-6) const {
        const double ex = extent(0);
        return std::fabs(ex - extent(1)) <= tol && std::fabs(ex - extent(2)) <= tol;
    }
    double volume() const {
        if (is_empty()) return 0.0;
        return extent(0) * extent(1) * extent(2);
    }

    // CLOSED-box intersection, and it is closed on purpose: a point exactly on
    // a shared face belongs to both boxes, so node culling cannot lose a point
    // that lies precisely on a query boundary.
    bool intersects(const BoundingBox& o) const {
        if (is_empty() || o.is_empty()) return false;
        for (int i = 0; i < 3; ++i)
            if (max[i] < o.min[i] || o.max[i] < min[i]) return false;
        return true;
    }

    bool contains(double x, double y, double z) const {
        return x >= min[0] && x <= max[0] && y >= min[1] && y <= max[1] &&
               z >= min[2] && z <= max[2];
    }
    bool contains(const RawPoint& p) const { return contains(p.x, p.y, p.z); }

    // Inset by `d` on every axis; used to build the region where a decoded
    // coordinate is UNAMBIGUOUSLY inside the query (see the render harness,
    // which separates the certain-interior from the boundary shell).
    BoundingBox inset(double d) const {
        BoundingBox b;
        for (int i = 0; i < 3; ++i) { b.min[i] = min[i] + d; b.max[i] = max[i] - d; }
        return b;
    }
    BoundingBox outset(double d) const {
        BoundingBox b;
        for (int i = 0; i < 3; ++i) { b.min[i] = min[i] - d; b.max[i] = max[i] + d; }
        return b;
    }
};

// ========================================================================
// ReaderDiagnostics - structural findings, separated by severity
// ========================================================================
// Split deliberately. `fatal` conditions stop the file being usable at all, and
// are what make open() fail. Everything else describes a file that can still be
// read but is not clean; a reader that refused such a file would be less useful
// than one that reports it, and a reader that silently accepted it would hide a
// writer bug. clean() covers both, so a test can assert on a self-built file.
struct ReaderDiagnostics {
    // ---- fatal ----
    bool     header_read = false;
    bool     magic_ok = false;
    bool     version_ok = false;
    bool     index_offset_sane = false;
    bool     directory_read = false;
    bool     table_fits = false;

    // ---- non-fatal, but a well-formed file has none of these ----
    uint64_t reserved_nonzero = 0;
    uint64_t trailing_bytes = 0;         // bytes after the node table
    uint64_t duplicate_node_ids = 0;
    uint64_t bad_chunk_length = 0;       // byte_length != point_count * 11
    uint64_t chunk_out_of_range = 0;     // chunk outside [64, index_offset)
    uint64_t overlapping_chunks = 0;     // two nodes claim the same bytes
    uint64_t degenerate_cells = 0;       // half_size <= 0
    uint64_t node_reserved_nonzero = 0;
    uint64_t envelope_not_cubic = 0;
    uint64_t nodes_outside_envelope = 0;

    bool fatal() const {
        return !(header_read && magic_ok && version_ok && index_offset_sane &&
                 directory_read && table_fits);
    }
    bool clean() const {
        return !fatal() && reserved_nonzero == 0 && trailing_bytes == 0 &&
               duplicate_node_ids == 0 && bad_chunk_length == 0 &&
               chunk_out_of_range == 0 && overlapping_chunks == 0 &&
               degenerate_cells == 0 && node_reserved_nonzero == 0 &&
               envelope_not_cubic == 0 && nodes_outside_envelope == 0;
    }
};

// ---- the inverse quantiser, in one place --------------------------------
// world = bounds_min + center - half_size + rel * 2 * half_size / 65535
//
// This is the spec's formula and it is the exact function LNTMath.h's
// node_local_to_world() implements, reused rather than re-derived so the writer,
// the reader and the Python reference cannot drift apart.
//
// THE BOUND IT GIVES. This inverse divides by 65535, and the writer's forward map
// (LNTMath.h quantize_to_16bit) ROUNDS onto those same 65535 intervals, so the
// two are exact inverses and the worst per-axis error is half an interval:
//
//     0.5 * node_cell_step()
//
// They were NOT always a matched pair: the forward map used to floor into 65536
// bins, which left a worst case of 65535/65536 = 0.99998 of a step and threw away
// half the payload's precision. The harness asserts the half-step bound over a
// dense sweep rather than trusting the arithmetic here.
inline RawPoint decode_point(const LNTPoint& p, const LNTNodeEntry& node,
                            const double origin[3]) {
    RawPoint out;
    out.x = node_local_to_world(p.x_rel, node.center[0], node.half_size, origin[0]);
    out.y = node_local_to_world(p.y_rel, node.center[1], node.half_size, origin[1]);
    out.z = node_local_to_world(p.z_rel, node.center[2], node.half_size, origin[2]);
    out.r = p.r;
    out.g = p.g;
    out.b = p.b;
    out.intensity = p.intensity;
    out.classification = p.classification;
    return out;
}

// Node cell as a world-space box: centre is a float32 OFFSET from bounds_min.
inline BoundingBox node_bounds(const LNTNodeEntry& node, const double origin[3]) {
    BoundingBox b;
    for (int i = 0; i < 3; ++i) {
        const double c = origin[i] + double(node.center[i]);
        b.min[i] = c - double(node.half_size);
        b.max[i] = c + double(node.half_size);
    }
    return b;
}

// A leaf node is exactly one with no children (the writer emits
// `children_mask = want_split ? mask : 0`), so this is a directory-only test
// with no payload read.
inline bool is_leaf_node(const LNTNodeEntry& node) {
    return node.children_mask == 0;
}

// ========================================================================
// Reader
// ========================================================================
class Reader {
public:
    Reader() = default;
    explicit Reader(const std::string& path) { open(path); }

    // ---- lifecycle -------------------------------------------------------
    // Reopens from scratch. Returns the same value as ok(). Never throws.
    bool open(const std::string& path);
    void close();

    bool ok() const { return ok_; }
    const std::string& error() const { return error_; }
    const std::string& path() const { return path_; }
    const std::string& last_error() const { return last_error_; }

    // ---- what the file says ---------------------------------------------
    const LNTHeader& header() const { return hdr_; }
    const LNTDirectoryHeader& directory() const { return dir_; }
    // LEAF points, which is the dataset size: LOD copies written for interior
    // nodes are NOT counted here (the writer records the input count).
    uint64_t get_total_points() const { return dir_.total_points; }
    uint64_t get_total_nodes() const { return dir_.total_nodes; }
    uint64_t get_file_bytes() const { return file_bytes_; }
    uint64_t get_index_offset() const { return hdr_.index_offset; }

    // The header's cubic envelope, and its centre - which is also the ideal
    // render origin for a float32 pipeline (see LNTVulkanEngine.h).
    BoundingBox get_bounds() const {
        return BoundingBox::from_min_max(hdr_.bounds_min[0], hdr_.bounds_min[1],
                                         hdr_.bounds_min[2], hdr_.bounds_max[0],
                                         hdr_.bounds_max[1], hdr_.bounds_max[2]);
    }
    double origin(int axis) const { return hdr_.bounds_min[axis]; }

    // ---- directory access ------------------------------------------------
    // Directory ORDER (the Morton order the writer emitted). Deterministic, so
    // queries and draw ranges built from it are reproducible.
    const std::vector<LNTNodeEntry>& nodes() const { return nodes_; }
    // The spec's lookup map, keyed by node_id. Both views are kept: the vector
    // gives stable ordering, the map gives O(1) id lookup, and the reader needs
    // each for a different job.
    const std::unordered_map<uint64_t, LNTNodeEntry>& node_index() const {
        return index_;
    }
    const LNTNodeEntry* find_node(uint64_t node_id) const {
        auto it = index_.find(node_id);
        return it == index_.end() ? nullptr : &it->second;
    }
    // A node's world-space cell, from the directory alone - no payload read.
    BoundingBox bounds_of(const LNTNodeEntry& node) const {
        return node_bounds(node, hdr_.bounds_min);
    }

    // ---- payload streaming ----------------------------------------------
    // Decodes `node`'s payload into world-space RawPoints. Reads only that
    // node's chunk. Returns false (and sets last_error()) for a node whose
    // chunk is inconsistent with the file - it never reads the header or the
    // trailer as if they were points, which a corrupt byte_offset would
    // otherwise make it do.
    bool read_node_points(const LNTNodeEntry& node, std::vector<RawPoint>& out_points);
    bool read_node_points(uint64_t node_id, std::vector<RawPoint>& out_points);
    // The raw payload, for a caller that wants to dequantise itself or compare
    // bytes without a float round-trip.
    bool read_node_raw(const LNTNodeEntry& node, std::vector<LNTPoint>& out_raw);

    // ---- spatial culling -------------------------------------------------
    // Every node whose cell INTERSECTS the box. Directory-only scan: no payload
    // is read, and the result is a superset of the nodes that actually hold a
    // point in the box (a cell can overlap the box and hold no point inside it).
    // That superset property is the useful one - it cannot MISS a node whose
    // points are in the box - and it is what a draw-range selection needs.
    std::vector<LNTNodeEntry> query_nodes_in_bounds(const BoundingBox& query_box) const;

    // The exact variants, for a caller that needs "only nodes that really do
    // contain a point inside the box" and is willing to pay a decode. Kept
    // separate from query_nodes_in_bounds so the cheap path stays cheap.
    std::vector<LNTNodeEntry> query_nodes_with_points_in_bounds(const BoundingBox& query_box);
    bool read_points_in_bounds(const BoundingBox& query_box,
                               std::vector<RawPoint>& out_points);

    // ---- structure ------------------------------------------------------
    ReaderDiagnostics diagnose() const;

private:
    bool fail(const std::string& msg);

    std::ifstream f_;
    std::string path_;
    std::string error_;
    std::string last_error_;

    LNTHeader hdr_{};
    LNTDirectoryHeader dir_{};
    std::vector<LNTNodeEntry> nodes_;
    std::unordered_map<uint64_t, LNTNodeEntry> index_;
    uint64_t file_bytes_ = 0;
    bool ok_ = false;
};

// ---- dataset-wide helpers, DEFINED IN LNTReader.cpp ---------------------
// The only non-inline reader functions. Declared here, defined in the split TU,
// because they are used by tools that always compile against LNTReader.cpp
// (lnt_view.cpp). main.cpp deliberately does not call them, so the
// single-command harness build still links with no second translation unit.

// Human-readable summary of an opened file: the validated header fields, the
// directory counts and the envelope - reported from what the reader actually
// parsed, never re-read.
std::string describe_file(const Reader& r);

// Decode every node, keeping leaves and interior LOD samples APART.
//
// The separation is not cosmetic. A leaf's chunk is the authoritative
// full-detail copy of its points; an interior node's chunk is a strided LOD
// sample that deliberately REPEATS points held by its descendants. Merging the
// two into one list would hand back the same point many times, so the counts
// are reported separately and `out_points` receives the leaf points only.
bool read_all_leaf_points(Reader& r, std::vector<RawPoint>& out_points,
                          uint64_t& leaf_points, uint64_t& lod_points,
                          uint64_t& leaf_nodes, uint64_t& lod_nodes);

// ========================================================================
// implementations
// ========================================================================
inline bool Reader::fail(const std::string& msg) {
    error_ = msg;
    ok_ = false;
    return false;
}

inline void Reader::close() {
    f_.close();
    nodes_.clear();
    index_.clear();
    hdr_ = LNTHeader{};
    dir_ = LNTDirectoryHeader{};
    file_bytes_ = 0;
    ok_ = false;
}

inline bool Reader::open(const std::string& path) {
    close();
    error_.clear();
    last_error_.clear();
    path_ = path;

    f_.open(path, std::ios::binary);
    if (!f_.good()) return fail("cannot open '" + path + "'");

    f_.seekg(0, std::ios::end);
    const std::streamoff end = f_.tellg();
    if (end < 0) return fail("cannot size '" + path + "'");
    file_bytes_ = static_cast<uint64_t>(end);
    if (file_bytes_ < kHeaderBytes)
        return fail("file is smaller than the 64-byte header");

    // ---- header ----------------------------------------------------------
    f_.seekg(0, std::ios::beg);
    f_.read(reinterpret_cast<char*>(&hdr_), sizeof(hdr_));
    if (f_.gcount() != static_cast<std::streamsize>(sizeof(hdr_)))
        return fail("short read of LNTHeader");
    if (!magic_matches(hdr_.magic))
        return fail(std::string("bad magic (expected 'LNT\\0', got '") +
                    std::string(hdr_.magic, hdr_.magic[3] == '\0' ? 3 : 4) + "')");
    if (hdr_.version != kFormatVersion)
        return fail("unsupported format version " +
                    std::to_string(hdr_.version) + " (this reader is version " +
                    std::to_string(kFormatVersion) + ")");

    // index_offset must address a whole directory header INSIDE the file. A
    // negative/overflowing value cannot occur (uint64) but a truncated or
    // hand-edited one can, and reading past EOF must be refused here rather
    // than discovered later as a zero-filled node table.
    if (hdr_.index_offset < kHeaderBytes ||
        hdr_.index_offset > file_bytes_ - kDirHeaderBytes)
        return fail("index_offset " + std::to_string(hdr_.index_offset) +
                    " does not address a directory header inside the file (" +
                    std::to_string(file_bytes_) + " bytes)");

    // ---- directory -------------------------------------------------------
    f_.clear();
    f_.seekg(static_cast<std::streamoff>(hdr_.index_offset), std::ios::beg);
    f_.read(reinterpret_cast<char*>(&dir_), sizeof(dir_));
    if (f_.gcount() != static_cast<std::streamsize>(sizeof(dir_)))
        return fail("short read of LNTDirectoryHeader");

    // The node table must fit. Checked with the multiplication guarded so a
    // hostile total_nodes cannot wrap and pass.
    const uint64_t table_bytes = dir_.total_nodes * kNodeEntryBytes;
    if (dir_.total_nodes != 0 &&
        table_bytes / kNodeEntryBytes != dir_.total_nodes)
        return fail("total_nodes is implausible: " + std::to_string(dir_.total_nodes));
    if (hdr_.index_offset + kDirHeaderBytes + table_bytes > file_bytes_)
        return fail("node table does not fit in the file (" +
                    std::to_string(dir_.total_nodes) + " nodes need " +
                    std::to_string(table_bytes) + " bytes)");

    nodes_.resize(static_cast<std::size_t>(dir_.total_nodes));
    if (!nodes_.empty()) {
        f_.read(reinterpret_cast<char*>(nodes_.data()),
                static_cast<std::streamsize>(table_bytes));
        if (f_.gcount() != static_cast<std::streamsize>(table_bytes))
            return fail("short read of the node table");
    }
    index_.reserve(nodes_.size() * 2);
    for (const LNTNodeEntry& e : nodes_) index_.emplace(e.node_id, e);

    ok_ = true;
    return true;
}

inline bool Reader::read_node_raw(const LNTNodeEntry& node,
                                  std::vector<LNTPoint>& out_raw) {
    out_raw.clear();
    last_error_.clear();
    if (!ok_) { last_error_ = "reader is not open"; return false; }

    // `byte_length` and `point_count` are redundant by design (one is the
    // other times 10); a file where they disagree cannot be decoded safely, so
    // it is refused rather than trusted.
    if (node.byte_length != static_cast<uint64_t>(node.point_count) * kPointBytes) {
        last_error_ = "node " + std::to_string(node.node_id) +
                      ": byte_length " + std::to_string(node.byte_length) +
                      " != point_count * 10 (" +
                      std::to_string(static_cast<uint64_t>(node.point_count) * kPointBytes) +
                      ")";
        return false;
    }
    if (node.point_count == 0) return true;   // legitimately empty node

    // The chunk must live in the PAYLOAD region only. Without this a corrupted
    // byte_offset (or index_offset) would happily decode the header or the node
    // table as points and produce plausible-looking garbage coordinates.
    const uint64_t lo = node.byte_offset;
    const uint64_t hi = lo + node.byte_length;
    if (lo < kHeaderBytes || hi > hdr_.index_offset || hi < lo) {
        last_error_ = "node " + std::to_string(node.node_id) +
                      ": chunk [" + std::to_string(lo) + ", " + std::to_string(hi) +
                      ") is outside the payload region [64, " +
                      std::to_string(hdr_.index_offset) + ")";
        return false;
    }

    out_raw.resize(node.point_count);
    f_.clear();
    f_.seekg(static_cast<std::streamoff>(lo), std::ios::beg);
    f_.read(reinterpret_cast<char*>(out_raw.data()),
            static_cast<std::streamsize>(node.byte_length));
    if (f_.gcount() != static_cast<std::streamsize>(node.byte_length)) {
        last_error_ = "node " + std::to_string(node.node_id) + ": short payload read";
        out_raw.clear();
        return false;
    }
    return true;
}

inline bool Reader::read_node_points(const LNTNodeEntry& node,
                                     std::vector<RawPoint>& out_points) {
    out_points.clear();
    std::vector<LNTPoint> raw;
    if (!read_node_raw(node, raw)) return false;
    out_points.resize(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i)
        out_points[i] = decode_point(raw[i], node, hdr_.bounds_min);
    return true;
}

inline bool Reader::read_node_points(uint64_t node_id,
                                     std::vector<RawPoint>& out_points) {
    const LNTNodeEntry* n = find_node(node_id);
    if (n == nullptr) {
        out_points.clear();
        last_error_ = "no node with id " + std::to_string(node_id);
        return false;
    }
    return read_node_points(*n, out_points);
}

inline std::vector<LNTNodeEntry>
Reader::query_nodes_in_bounds(const BoundingBox& query_box) const {
    std::vector<LNTNodeEntry> out;
    if (!ok_ || query_box.is_empty()) return out;
    out.reserve(nodes_.size());
    for (const LNTNodeEntry& e : nodes_)
        if (node_bounds(e, hdr_.bounds_min).intersects(query_box))
            out.push_back(e);
    return out;
}

inline std::vector<LNTNodeEntry>
Reader::query_nodes_with_points_in_bounds(const BoundingBox& query_box) {
    std::vector<LNTNodeEntry> out;
    if (!ok_ || query_box.is_empty()) return out;
    std::vector<RawPoint> pts;
    for (const LNTNodeEntry& e : nodes_) {
        if (!node_bounds(e, hdr_.bounds_min).intersects(query_box)) continue;
        if (!read_node_points(e, pts)) continue;
        for (const RawPoint& p : pts)
            if (query_box.contains(p)) { out.push_back(e); break; }
    }
    return out;
}

inline bool Reader::read_points_in_bounds(const BoundingBox& query_box,
                                          std::vector<RawPoint>& out_points) {
    out_points.clear();
    last_error_.clear();
    if (!ok_) { last_error_ = "reader is not open"; return false; }
    if (query_box.is_empty()) return true;
    std::vector<RawPoint> pts;
    for (const LNTNodeEntry& e : nodes_) {
        if (!node_bounds(e, hdr_.bounds_min).intersects(query_box)) continue;
        if (!read_node_points(e, pts)) return false;
        for (const RawPoint& p : pts)
            if (query_box.contains(p)) out_points.push_back(p);
    }
    return true;
}

inline ReaderDiagnostics Reader::diagnose() const {
    ReaderDiagnostics d;
    if (!ok_) return d;
    d.header_read = true;
    d.magic_ok = magic_matches(hdr_.magic);
    d.version_ok = hdr_.version == kFormatVersion;
    d.index_offset_sane = hdr_.index_offset >= kHeaderBytes &&
                          hdr_.index_offset <= file_bytes_;
    d.directory_read = true;
    d.reserved_nonzero = hdr_.reserved != 0 ? 1 : 0;
    d.envelope_not_cubic = get_bounds().is_cube(1e-3) ? 0 : 1;

    const uint64_t declared = hdr_.index_offset + kDirHeaderBytes +
                              dir_.total_nodes * kNodeEntryBytes;
    d.table_fits = declared <= file_bytes_;
    d.trailing_bytes = declared <= file_bytes_ ? file_bytes_ - declared : 0;

    // Duplicate ids and chunk sanity.
    std::unordered_map<uint64_t, std::size_t> seen;
    seen.reserve(nodes_.size() * 2);
    for (const LNTNodeEntry& e : nodes_)
        if (!seen.emplace(e.node_id, 1).second) ++d.duplicate_node_ids;

    std::vector<std::pair<uint64_t, uint64_t>> spans;
    spans.reserve(nodes_.size());
    for (const LNTNodeEntry& e : nodes_) {
        if (e.byte_length != static_cast<uint64_t>(e.point_count) * kPointBytes)
            ++d.bad_chunk_length;
        const uint64_t lo = e.byte_offset;
        const uint64_t hi = lo + e.byte_length;
        if (lo < kHeaderBytes || hi > hdr_.index_offset || hi < lo)
            ++d.chunk_out_of_range;
        else if (e.byte_length > 0)
            spans.emplace_back(lo, hi);
        if (!(e.half_size > 0.0f)) ++d.degenerate_cells;
        for (int i = 0; i < 10; ++i)
            if (e.reserved[i] != 0) { ++d.node_reserved_nonzero; break; }
        // Every node cell must lie inside the advertised envelope, otherwise a
        // cull against the envelope could discard real geometry.
        const BoundingBox nb = node_bounds(e, hdr_.bounds_min);
        const BoundingBox env = get_bounds();
        const double tol = 1e-3;
        for (int a = 0; a < 3; ++a)
            if (nb.min[a] < env.min[a] - tol || nb.max[a] > env.max[a] + tol) {
                ++d.nodes_outside_envelope;
                break;
            }
    }
    std::sort(spans.begin(), spans.end());
    for (std::size_t i = 1; i < spans.size(); ++i)
        if (spans[i].first < spans[i - 1].second) { ++d.overlapping_chunks; break; }
    return d;
}

}  // namespace lnt

// The brief spells the class `LNT::Reader`; the rest of this tree uses the
// `lnt` namespace. An alias keeps one namespace while making both spellings
// name the same type, so no call site has to care.
namespace LNT = lnt;

#endif  // LNT_READER_H
