#pragma once
// LNTSurfaceLod.h - persistent, viewport-dependent surface LOD for native .lnt Shading
// (the compact crisp, multi-class, class-coloured path).
//
// TILES. A tile is a cell of an XY quadtree laid over the file's cubic envelope, so a
// tile at depth d has exactly the plan footprint of the octree nodes at tree_level d.
// Its points are those of the octree nodes at level d in its column (interior nodes
// carry the builder's strided representatives of their descendants; leaf nodes carry
// every record), plus the records of shallower leaves that fall inside the cell. A
// tile whose column holds only leaves is EXACT: it contains every original record of
// its cell, so it is the full-resolution surface there and is never refined further.
// Coarser tiles represent the records below them through the hierarchy; they never
// replace or delete a record - the file is unchanged and exact tiles stay available.
//
// SEAMS. A tile is triangulated with the halo of its eight same-level neighbours and
// keeps the triangles it OWNS (lowest tile index among a triangle's vertices), exactly
// the rule the whole-cut mesher uses, so same-level neighbours meet without a gap or an
// overlap. Each tile also keeps, per lower-numbered neighbour, the triangles of its own
// Delaunay that that neighbour owns ("fringe"). The fringe toward a neighbour is drawn
// only when that neighbour is not drawn - i.e. where a coarser or finer tile replaces
// it - so a mixed-LOD border is closed by a one-triangle overlap instead of a crack.
//
// IDENTITY. A tile is (depth, ix, iy) - independent of the camera - and is triangulated
// once per (original data, geometry settings). Results persist in the .lnt archive
// (LNTSurfaceArchive.h); a pan or zoom only selects, loads and uploads tiles.
//
// SELECTION. A tile is refined while its representative spacing projects to more than
// `tau_px` pixels. A refinement is shown only when every visible child is resident on
// the GPU (otherwise the parent stays): coverage never has holes while finer data
// arrives. Requests go to a worker pool, coarse levels first.
#include "LNTCamera.h"
#include "LNTReader.h"
#include "LNTSurfaceArchive.h"
#include "LNTSurfaceMesh.h"
#include "LNTVulkanEngine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lnt { namespace surface_lod {
using surface_archive::Digest;

constexpr uint32_t kAlgorithm = 3; // highest visible return at coincident XY
constexpr uint32_t kFlagExact = 1;
constexpr unsigned kMaxDepth = 27;

inline uint64_t make_key(unsigned d, uint32_t ix, uint32_t iy) {
    return (uint64_t(d) << 56) | (uint64_t(iy & 0xfffffffu) << 28) | uint64_t(ix & 0xfffffffu);
}
inline unsigned key_level(uint64_t k) { return unsigned(k >> 56); }
inline uint32_t key_x(uint64_t k) { return uint32_t(k & 0xfffffffu); }
inline uint32_t key_y(uint64_t k) { return uint32_t((k >> 28) & 0xfffffffu); }

// ---------------------------------------------------------------------------------
// Quadtree over the octree directory (immutable after build; safe to share).
// ---------------------------------------------------------------------------------
class Quadtree {
public:
    bool build(const Reader& rd) {
        collection_ = rd.is_collection();
        total_leaf_points_ = 0;
        if (collection_) return build_collection(rd);
        nodes_ = &rd.nodes();
        for (int a = 0; a < 3; ++a) origin_[a] = rd.origin(a);
        side_ = rd.header().bounds_max[0] - rd.header().bounds_min[0];
        if (!(side_ > 0.0) || nodes_->empty()) return false;
        at_.clear(); inside_.clear();
        max_level_ = 0;
        double mn[2] = {1e300, 1e300}, mx[2] = {-1e300, -1e300};
        for (uint32_t i = 0; i < nodes_->size(); ++i) {
            const LNTNodeEntry& n = (*nodes_)[i];
            const unsigned l = std::min<unsigned>(n.tree_level, kMaxDepth);
            max_level_ = std::max(max_level_, l);
            const uint32_t ix = cell_of_offset(n.center[0], l), iy = cell_of_offset(n.center[1], l);
            at_[make_key(l, ix, iy)].push_back(i);
            if (is_leaf_node(n)) {
                for (unsigned d = 0; d <= l; ++d)
                    inside_[make_key(d, ix >> (l - d), iy >> (l - d))] += n.point_count;
                total_leaf_points_ += n.point_count;
                const BoundingBox b = node_bounds(n, origin_);
                for (int a = 0; a < 2; ++a) { mn[a] = std::min(mn[a], b.min[a]); mx[a] = std::max(mx[a], b.max[a]); }
            }
        }
        xy_extent_ = std::max(mx[0] - mn[0], mx[1] - mn[1]);
        if (!(xy_extent_ > 0.0)) xy_extent_ = side_;
        return true;
    }
    const std::vector<LNTNodeEntry>& nodes() const { return *nodes_; }
    double side() const { return side_; }
    const double* origin() const { return origin_; }
    unsigned max_level() const { return max_level_; }
    double xy_extent() const { return xy_extent_; }
    uint64_t total_leaf_points() const { return total_leaf_points_; }

    // Cell index of a world coordinate at depth d. One division then an exact
    // power-of-two scale, so the cells of consecutive depths nest exactly.
    uint32_t cell_of(double w, int axis, unsigned d) const {
        const double t = (w - origin_[axis]) / side_;
        const double c = std::floor(t * double(uint64_t(1) << d));
        const double hi = double((uint64_t(1) << d) - 1);
        return uint32_t(std::min(std::max(c, 0.0), hi));
    }
    void cell_xy(uint64_t k, double mn[2], double mx[2]) const {
        const double cell = side_ / double(uint64_t(1) << key_level(k));
        mn[0] = origin_[0] + cell * key_x(k); mx[0] = mn[0] + cell;
        mn[1] = origin_[1] + cell * key_y(k); mx[1] = mn[1] + cell;
    }
    double cell_size(unsigned d) const { return side_ / double(uint64_t(1) << d); }
    // The depth-d cell of a record of node `idx` (tree_level <= d), clamped into that
    // node's own footprint: a record on (or float-rounded across) the node's border
    // stays in a cell that gathers this node, so no record is ever lost or doubled.
    void record_cell(uint32_t idx, double x, double y, unsigned d, uint32_t& ix, uint32_t& iy) const {
        if (collection_) { ix = cell_of(x, 0, d); iy = cell_of(y, 1, d); return; }
        const LNTNodeEntry& n = (*nodes_)[idx];
        const unsigned l = std::min<unsigned>(n.tree_level, d);
        const uint64_t span = uint64_t(1) << (d - l);
        const uint64_t lx = uint64_t(cell_of_offset(n.center[0], l)) * span, ly = uint64_t(cell_of_offset(n.center[1], l)) * span;
        ix = uint32_t(std::min<uint64_t>(std::max<uint64_t>(cell_of(x, 0, d), lx), lx + span - 1));
        iy = uint32_t(std::min<uint64_t>(std::max<uint64_t>(cell_of(y, 1, d), ly), ly + span - 1));
    }
    const std::vector<uint32_t>* column(uint64_t k) const {
        const auto it = at_.find(k);
        return it == at_.end() ? nullptr : &it->second;
    }
    // Leaves at depth < d whose cell contains tile k (their records reach into it).
    void shallow_leaves(uint64_t k, std::vector<uint32_t>& out) const {
        out.clear();
        const unsigned d = key_level(k);
        for (unsigned l = 0; l < d; ++l) {
            const auto* c = column(make_key(l, key_x(k) >> (d - l), key_y(k) >> (d - l)));
            if (!c) continue;
            for (uint32_t i : *c) if (is_leaf_node((*nodes_)[i])) out.push_back(i);
        }
    }
    bool exists(uint64_t k) const {
        if (column(k)) return true;
        std::vector<uint32_t> s;
        shallow_leaves(k, s);
        return !s.empty();
    }
    bool exact(uint64_t k) const {
        if (const auto* c = column(k))
            for (uint32_t i : *c) if (!is_leaf_node((*nodes_)[i])) return false;
        return true;
    }
    void children(uint64_t k, std::vector<uint64_t>& out) const {
        out.clear();
        const unsigned d = key_level(k);
        if (d >= kMaxDepth || exact(k)) return;
        bool covered = false;                       // a leaf at depth <= d reaches every child
        if (const auto* c = column(k))
            for (uint32_t i : *c) covered |= is_leaf_node((*nodes_)[i]);
        if (!covered) { std::vector<uint32_t> s; shallow_leaves(k, s); covered = !s.empty(); }
        for (uint32_t j = 0; j < 4; ++j) {
            const uint64_t c = make_key(d + 1, key_x(k) * 2 + (j & 1), key_y(k) * 2 + (j >> 1));
            if (covered || column(c)) out.push_back(c);
        }
    }
    // Original leaf records of leaves at depth >= d that lie inside tile k.
    uint64_t leaf_points_inside(uint64_t k) const {
        const auto it = inside_.find(k);
        return it == inside_.end() ? 0 : it->second;
    }
    // Representatives a tile will hold (directory estimate, for the refinement metric).
    double estimated_points(uint64_t k) const {
        double n = 0.0;
        if (const auto* c = column(k)) for (uint32_t i : *c) n += (*nodes_)[i].point_count;
        std::vector<uint32_t> s;
        shallow_leaves(k, s);
        const double area = std::pow(cell_size(key_level(k)), 2);
        for (uint32_t i : s) {
            const double h = 2.0 * (*nodes_)[i].half_size;
            n += (*nodes_)[i].point_count * std::min(1.0, area / std::max(h * h, 1e-12));
        }
        return n;
    }
    // Conservative Z range of a tile from its node cells.
    void z_range(uint64_t k, double& z0, double& z1) const {
        z0 = 1e300; z1 = -1e300;
        auto add = [&](uint32_t i) {
            const BoundingBox b = node_bounds((*nodes_)[i], origin_);
            z0 = std::min(z0, b.min[2]); z1 = std::max(z1, b.max[2]);
        };
        if (const auto* c = column(k)) for (uint32_t i : *c) add(i);
        std::vector<uint32_t> s;
        shallow_leaves(k, s);
        for (uint32_t i : s) add(i);
        if (z0 > z1) { z0 = origin_[2]; z1 = origin_[2] + side_; }
    }

private:
    bool collection_ = false;
    bool build_collection(const Reader& rd) {
        nodes_ = &rd.nodes();
        for (int a=0;a<3;++a) origin_[a] = rd.origin(a);
        side_ = rd.header().bounds_max[0] - rd.header().bounds_min[0];
        if (!(side_ > 0.0) || nodes_->empty()) return false;
        at_.clear(); inside_.clear(); max_level_ = 0;
        std::unordered_map<uint64_t,unsigned> resolution;
        for (const auto& n : *nodes_) {
            const double ratio = side_ / std::max(2.0 * double(n.half_size), 1e-12);
            resolution[n.node_id] = unsigned(std::max(0.0, std::min(double(kMaxDepth), std::floor(std::log2(ratio) + 1e-9))));
        }
        auto cells = [&](const BoundingBox& b, unsigned d, auto emit) {
            const uint32_t x0=cell_of(b.min[0],0,d), y0=cell_of(b.min[1],1,d);
            const uint32_t x1=cell_of(std::nextafter(b.max[0],-HUGE_VAL),0,d);
            const uint32_t y1=cell_of(std::nextafter(b.max[1],-HUGE_VAL),1,d);
            for(uint32_t y=y0;y<=y1;++y) for(uint32_t x=x0;x<=x1;++x) emit(make_key(d,x,y));
        };
        double lo[2]={1e300,1e300}, hi[2]={-1e300,-1e300};
        for (uint32_t i=0;i<nodes_->size();++i) {
            const auto& n=(*nodes_)[i];
            const auto parent=resolution.find(n.node_id >> 3);
            const unsigned first = parent==resolution.end() ? 0 : std::min(kMaxDepth,parent->second+1);
            const unsigned last = is_leaf_node(n) ? first : std::max(first,resolution[n.node_id]);
            max_level_=std::max(max_level_,last);
            const BoundingBox box=node_bounds(n,origin_);
            for(unsigned d=first;d<=last;++d) cells(box,d,[&](uint64_t k){at_[k].push_back(i);});
            if(is_leaf_node(n)) {
                total_leaf_points_+=n.point_count;
                for(int a=0;a<2;++a){lo[a]=std::min(lo[a],box.min[a]);hi[a]=std::max(hi[a],box.max[a]);}
                for(unsigned d=0;d<=first;++d) {
                    // Directory-only estimates for coarse cuts. Exact builds
                    // replace these estimates with the actual gathered count.
                    uint64_t assigned=0, first_key=0; bool have=false;
                    cells(box,d,[&](uint64_t k){
                        if(!have){first_key=k;have=true;}
                        double mn[2],mx[2];cell_xy(k,mn,mx);
                        const double ax=std::max(0.0,std::min(mx[0],box.max[0])-std::max(mn[0],box.min[0]));
                        const double ay=std::max(0.0,std::min(mx[1],box.max[1])-std::max(mn[1],box.min[1]));
                        const double full=std::max(box.extent(0)*box.extent(1),1e-24);
                        const uint64_t count=std::min<uint64_t>(n.point_count-assigned,uint64_t(n.point_count*(ax*ay/full)));
                        inside_[k]+=count;assigned+=count;
                    });
                    if(have)inside_[first_key]+=n.point_count-assigned;
                }
            }
        }
        xy_extent_=std::max(hi[0]-lo[0],hi[1]-lo[1]);
        if(!(xy_extent_>0))xy_extent_=side_;
        return true;
    }
    uint32_t cell_of_offset(float offset, unsigned l) const {
        const double c = std::floor(double(offset) / side_ * double(uint64_t(1) << l));
        return uint32_t(std::min(std::max(c, 0.0), double((uint64_t(1) << l) - 1)));
    }
    const std::vector<LNTNodeEntry>* nodes_ = nullptr;
    double origin_[3]{}, side_ = 0.0, xy_extent_ = 0.0;
    unsigned max_level_ = 0;
    uint64_t total_leaf_points_ = 0;
    std::unordered_map<uint64_t, std::vector<uint32_t>> at_;
    std::unordered_map<uint64_t, uint64_t> inside_;
};

// ---------------------------------------------------------------------------------
// A class edit of one stored record (a leaf record or one of its LOD copies): the chunk
// and index it lives at, its exact decoded position (bit-for-bit what the point reader
// and the tile vertices hold), and the class before / after.
// ---------------------------------------------------------------------------------
struct ClassEdit {
    uint32_t node = 0, local = 0;
    double x = 0.0, y = 0.0, z = 0.0;
    uint8_t cls = 0, old_cls = 0;
};

// ---------------------------------------------------------------------------------
// Decoded node records, shared by the build workers (LRU by bytes).
// ---------------------------------------------------------------------------------
class NodePoints {
public:
    using Ptr = std::shared_ptr<const std::vector<SurfacePoint>>;
    explicit NodePoints(uint64_t budget) : budget_(budget) {}
    void clear(uint64_t generation) {
        std::lock_guard<std::mutex> lk(mu_);
        map_.clear(); bytes_ = 0; gen_ = generation;
    }
    Ptr get(uint32_t idx, Reader& rd, const std::vector<LNTNodeEntry>& nodes, uint64_t generation,
            std::shared_mutex* io_lock, std::string* error = nullptr) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (generation != gen_) { if(error)*error="tile source generation changed"; return nullptr; }
            auto it = map_.find(idx);
            if (it != map_.end()) { it->second.tick = ++tick_; return it->second.ptr; }
        }
        std::vector<RawPoint> raw;
        {
            std::shared_lock<std::shared_mutex> lk;
            if (io_lock) lk = std::shared_lock<std::shared_mutex>(*io_lock);
            if (!rd.read_node_points(nodes[idx], raw)) {
                if(error)*error="tile source node "+std::to_string(nodes[idx].node_id)+": "+rd.last_error();
                return nullptr;
            }
            if(raw.size()!=nodes[idx].point_count) {
                if(error)*error="tile source node "+std::to_string(nodes[idx].node_id)+": record count mismatch (expected "+std::to_string(nodes[idx].point_count)+", read "+std::to_string(raw.size())+")";
                return nullptr;
            }
        }
        auto v = std::make_shared<std::vector<SurfacePoint>>();
        v->reserve(raw.size());
        for (const RawPoint& p : raw) v->push_back({p.x, p.y, p.z, p.classification, p.r, p.g, p.b, p.intensity});
        std::lock_guard<std::mutex> lk(mu_);
        if (generation != gen_) return nullptr;
        auto ov = overrides_.find(idx);                   // edits not yet in the file
        if (ov != overrides_.end())
            for (const auto& e : ov->second) if (e.first < v->size()) (*v)[e.first].cls = e.second;
        const uint64_t b = v->size() * sizeof(SurfacePoint);
        while (bytes_ + b > budget_ && !map_.empty()) {
            auto victim = map_.begin();
            for (auto it = map_.begin(); it != map_.end(); ++it) if (it->second.tick < victim->second.tick) victim = it;
            bytes_ -= victim->second.bytes;
            map_.erase(victim);
        }
        map_[idx] = {v, ++tick_, b};
        bytes_ += b;
        return v;
    }
    uint64_t bytes() const { std::lock_guard<std::mutex> lk(mu_); return bytes_; }
    // Class edits: cached records are patched copy-on-write (a worker holding the old
    // vector keeps it) and the cache moves to `generation`. With `pending` the edits
    // are also applied to records read later, until clear_overrides() (the file holds
    // them by then).
    void apply_edits(uint64_t generation, const std::vector<ClassEdit>& edits, bool pending) {
        std::lock_guard<std::mutex> lk(mu_);
        gen_ = generation;
        std::unordered_map<uint32_t, std::vector<const ClassEdit*>> by_node;
        for (const auto& e : edits) by_node[e.node].push_back(&e);
        for (const auto& kv : by_node) {
            if (pending) for (const ClassEdit* e : kv.second) overrides_[kv.first][e->local] = e->cls;
            auto it = map_.find(kv.first);
            if (it == map_.end()) continue;
            auto copy = it->second.ptr.use_count() == 1
                ? std::const_pointer_cast<std::vector<SurfacePoint>>(it->second.ptr)
                : std::make_shared<std::vector<SurfacePoint>>(*it->second.ptr);
            for (const ClassEdit* e : kv.second) if (e->local < copy->size()) (*copy)[e->local].cls = e->cls;
            it->second.ptr = copy;
        }
    }
    void clear_overrides() { std::lock_guard<std::mutex> lk(mu_); overrides_.clear(); }
private:
    struct Entry { Ptr ptr; uint64_t tick = 0, bytes = 0; };
    mutable std::mutex mu_;
    std::unordered_map<uint32_t, Entry> map_;
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint8_t>> overrides_;
    uint64_t budget_, bytes_ = 0, tick_ = 0, gen_ = 0;
};

// ---------------------------------------------------------------------------------
// One tile's geometry.
// ---------------------------------------------------------------------------------
struct TileData {
    uint64_t key = 0;
    uint32_t flags = 0;
    uint32_t groups[5]{};          // triangles: owned, then fringe owned by W, SW, S, SE
    uint64_t source_points = 0;    // original leaf records inside the cell (any class)
    uint32_t represented = 0;      // visible records (exact) / representatives (coarse) used
    uint32_t referenced = 0;       // ...of which an owned triangle references
    double bmin[3]{}, bmax[3]{};
    std::vector<double> positions;
    std::vector<uint8_t> vertex_class;
    std::vector<int32_t> faces;            // LOCAL vertex indices
    std::vector<double> coverage_positions; // represented records no owned triangle references
    std::vector<uint8_t> coverage_class;
    // Build-time only, released after encoding: the source node of every vertex and
    // coverage record. The stored record keeps each position as that node's 16-bit
    // coordinates, which decode bit-for-bit to the same double the point reader yields.
    std::vector<uint32_t> vertex_node, coverage_node;
    uint32_t vertices() const { return uint32_t(vertex_class.size()); }
    uint32_t triangles() const { return uint32_t(faces.size() / 3); }
    uint32_t coverage() const { return uint32_t(coverage_class.size()); }
    bool exact() const { return (flags & kFlagExact) != 0; }
    bool empty() const { return faces.empty() && coverage_class.empty(); }
    uint64_t bytes() const {
        return positions.size() * 8 + vertex_class.size() + faces.size() * 4 + coverage_positions.size() * 8 +
               coverage_class.size() + (vertex_node.size() + coverage_node.size()) * 4 + sizeof(TileData);
    }
};

constexpr uint32_t kPayloadVersion = 2;
struct PayloadHeader {
    char magic[4] = {'L', 'N', 'T', 'P'};
    uint32_t version = kPayloadVersion;
    uint64_t key = 0, source_points = 0;
    uint32_t flags = 0, vertices = 0;
    uint32_t groups[5]{};
    uint32_t coverage = 0, represented = 0, referenced = 0;
    uint32_t node_count = 0;       // > 0: positions are node-relative 16-bit coordinates
    uint32_t reserved = 0;
    double bmin[3]{}, bmax[3]{};
};
static_assert(sizeof(PayloadHeader) == 120, "tile payload header layout changed");

namespace codec_detail {
// The 16-bit coordinate whose decode is bit-identical to `xyz` in node `n`, or false.
inline bool node_rel(const LNTNodeEntry& n, const double* origin, const double* xyz, uint16_t rel[3]) {
    for (int a = 0; a < 3; ++a) {
        const double lo = origin[a] + double(n.center[a]) - double(n.half_size);
        const double hi = origin[a] + double(n.center[a]) + double(n.half_size);
        const int q = quantize_to_16bit(xyz[a], lo, hi);
        bool ok = false;
        for (int step : {0, -1, 1}) {
            const int c = q + step;
            if (c < 0 || c > kLocalMax) continue;
            const double back = node_local_to_world(uint32_t(c), n.center[a], n.half_size, origin[a]);
            if (std::memcmp(&back, &xyz[a], sizeof(double)) == 0) { rel[a] = uint16_t(c); ok = true; break; }
        }
        if (!ok) return false;
    }
    return true;
}
inline uint32_t zigzag(int32_t v) { return (uint32_t(v) << 1) ^ uint32_t(v >> 31); }
inline int32_t unzigzag(uint32_t v) { return int32_t(v >> 1) ^ -int32_t(v & 1u); }

// Node-relative position stream: node refs, then x / y / z planes, each byte-shuffled.
struct RelStream {
    std::vector<uint16_t> ref, rel[3];
};
inline bool build_rel(const std::vector<double>& pos, const std::vector<uint32_t>& node,
                      const std::vector<LNTNodeEntry>& nodes, const double* origin,
                      std::unordered_map<uint32_t, uint16_t>& table, std::vector<uint32_t>& order, RelStream& out) {
    const size_t n = pos.size() / 3;
    if (node.size() != n) return false;
    out.ref.resize(n);
    for (auto& r : out.rel) r.resize(n);
    for (size_t i = 0; i < n; ++i) {
        if (node[i] >= nodes.size()) return false;
        auto it = table.find(node[i]);
        if (it == table.end()) {
            if (order.size() >= 65535) return false;
            it = table.emplace(node[i], uint16_t(order.size())).first;
            order.push_back(node[i]);
        }
        uint16_t rel[3];
        if (!node_rel(nodes[node[i]], origin, &pos[i * 3], rel)) return false;
        out.ref[i] = it->second;
        for (int a = 0; a < 3; ++a) out.rel[a][i] = rel[a];
    }
    return true;
}
}  // namespace codec_detail

// Lossless record body. Positions are stored as (source node, 16-bit node coordinate),
// verified at encode time to decode to the identical doubles (else raw doubles are kept);
// faces as zigzag deltas; every array byte-plane shuffled, then zstd.
inline bool encode_tile(const TileData& t, const Digest& settings, const Quadtree& qt,
                        embedded_surface::MeshCodec& codec, surface_archive::PendingRecord& rec,
                        bool* node_relative = nullptr) {
    using namespace codec_detail;
    PayloadHeader h;
    h.key = t.key; h.source_points = t.source_points; h.flags = t.flags; h.vertices = t.vertices();
    for (int i = 0; i < 5; ++i) h.groups[i] = t.groups[i];
    h.coverage = t.coverage(); h.represented = t.represented; h.referenced = t.referenced;
    for (int a = 0; a < 3; ++a) { h.bmin[a] = t.bmin[a]; h.bmax[a] = t.bmax[a]; }
    std::unordered_map<uint32_t, uint16_t> table;
    std::vector<uint32_t> order;
    RelStream vs, cs;
    const bool rel = build_rel(t.positions, t.vertex_node, qt.nodes(), qt.origin(), table, order, vs) &&
                     build_rel(t.coverage_positions, t.coverage_node, qt.nodes(), qt.origin(), table, order, cs);
    h.node_count = rel ? uint32_t(order.size()) : 0;
    if (rel && order.empty()) h.node_count = 0;
    if (node_relative) *node_relative = h.node_count > 0 || (rel && t.vertices() == 0 && t.coverage() == 0);
    const size_t V = t.vertices(), C = t.coverage(), F = t.faces.size();
    const size_t pos_bytes = h.node_count ? 8 : 24;
    std::vector<char> raw(sizeof(h) + h.node_count * 4 + V * (pos_bytes + 1) + F * 4 + C * (pos_bytes + 1));
    char* p = raw.data();
    std::memcpy(p, &h, sizeof(h)); p += sizeof(h);
    auto put16 = [&](const std::vector<uint16_t>& v) {
        embedded_surface::shuffle_words<2, true>(p, reinterpret_cast<const char*>(v.data()), v.size() * 2);
        p += v.size() * 2;
    };
    auto put_positions = [&](const std::vector<double>& pos, const RelStream& s) {
        if (h.node_count) { put16(s.ref); for (const auto& r : s.rel) put16(r); }
        else {
            embedded_surface::shuffle_words<8, true>(p, reinterpret_cast<const char*>(pos.data()), pos.size() * 8);
            p += pos.size() * 8;
        }
    };
    if (h.node_count) { std::memcpy(p, order.data(), order.size() * 4); p += order.size() * 4; }
    put_positions(t.positions, vs);
    if (V) std::memcpy(p, t.vertex_class.data(), V);
    p += V;
    {
        std::vector<uint32_t> d(F);
        int32_t prev = 0;
        for (size_t i = 0; i < F; ++i) { d[i] = zigzag(t.faces[i] - prev); prev = t.faces[i]; }
        embedded_surface::shuffle_words<4, true>(p, reinterpret_cast<const char*>(d.data()), F * 4);
        p += F * 4;
    }
    put_positions(t.coverage_positions, cs);
    if (C) std::memcpy(p, t.coverage_class.data(), C);
    rec = surface_archive::PendingRecord{};
    rec.settings = settings;
    auto& e = rec.meta;
    e.key = t.key; e.raw = raw.size(); e.source_points = t.source_points;
    e.crc_raw = surface_archive::crc32c(raw.data(), raw.size());
    e.vertices = t.vertices(); e.triangles = t.triangles(); e.coverage = t.coverage();
    e.represented = t.represented; e.flags = uint16_t(t.flags);
    for (int a = 0; a < 3; ++a) { e.bmin[a] = t.bmin[a]; e.bmax[a] = t.bmax[a]; }
    if (codec.encode(raw.data(), raw.size(), rec.stored, 3)) e.codec = 1;
    else { rec.stored.swap(raw); e.codec = 0; }
    e.stored = rec.stored.size();
    e.crc_stored = surface_archive::crc32c(rec.stored.data(), rec.stored.size());
    return true;
}

inline bool decode_tile(const std::vector<char>& raw, uint64_t key, const Quadtree& qt, TileData& t, std::string* err) {
    using namespace codec_detail;
    auto fail = [&](const char* why) { if (err) *err = why; return false; };
    if (raw.size() < sizeof(PayloadHeader)) return fail("tile payload truncated");
    PayloadHeader h;
    std::memcpy(&h, raw.data(), sizeof(h));
    if (std::memcmp(h.magic, "LNTP", 4) != 0 || h.version != kPayloadVersion || h.key != key || h.reserved)
        return fail("tile payload header mismatch");
    uint64_t tris = 0;
    for (int i = 0; i < 5; ++i) tris += h.groups[i];
    const uint64_t pos_bytes = h.node_count ? 8 : 24;
    const uint64_t need = sizeof(h) + uint64_t(h.node_count) * 4 + uint64_t(h.vertices) * (pos_bytes + 1) +
                          tris * 12 + uint64_t(h.coverage) * (pos_bytes + 1);
    if (need != raw.size()) return fail("tile payload size mismatch");
    if (h.referenced > h.represented || uint64_t(h.represented) - h.referenced != h.coverage) return fail("tile accounting mismatch");
    t = TileData{};
    t.key = key; t.flags = h.flags; t.source_points = h.source_points;
    for (int i = 0; i < 5; ++i) t.groups[i] = h.groups[i];
    t.represented = h.represented; t.referenced = h.referenced;
    for (int a = 0; a < 3; ++a) { t.bmin[a] = h.bmin[a]; t.bmax[a] = h.bmax[a]; }
    const char* p = raw.data() + sizeof(h);
    std::vector<uint32_t> table(h.node_count);
    if (h.node_count) { std::memcpy(table.data(), p, table.size() * 4); p += table.size() * 4; }
    for (uint32_t n : table) if (n >= qt.nodes().size()) return fail("tile node reference out of range");
    auto get16 = [&](std::vector<uint16_t>& v, size_t n) {
        v.resize(n);
        embedded_surface::shuffle_words<2, false>(reinterpret_cast<char*>(v.data()), p, n * 2);
        p += n * 2;
    };
    auto get_positions = [&](std::vector<double>& pos, size_t n) -> bool {
        pos.resize(n * 3);
        if (!h.node_count) {
            embedded_surface::shuffle_words<8, false>(reinterpret_cast<char*>(pos.data()), p, n * 24);
            p += n * 24;
            return true;
        }
        std::vector<uint16_t> ref, rel[3];
        get16(ref, n);
        for (auto& r : rel) get16(r, n);
        const double* o = qt.origin();
        for (size_t i = 0; i < n; ++i) {
            if (ref[i] >= table.size()) return false;
            const LNTNodeEntry& e = qt.nodes()[table[ref[i]]];
            for (int a = 0; a < 3; ++a) pos[i * 3 + a] = node_local_to_world(rel[a][i], e.center[a], e.half_size, o[a]);
        }
        return true;
    };
    if (!get_positions(t.positions, h.vertices)) return fail("tile node reference out of range");
    t.vertex_class.assign(reinterpret_cast<const uint8_t*>(p), reinterpret_cast<const uint8_t*>(p) + h.vertices);
    p += h.vertices;
    {
        std::vector<uint32_t> d(size_t(tris) * 3);
        embedded_surface::shuffle_words<4, false>(reinterpret_cast<char*>(d.data()), p, d.size() * 4);
        p += d.size() * 4;
        t.faces.resize(d.size());
        int32_t prev = 0;
        for (size_t i = 0; i < d.size(); ++i) { prev += unzigzag(d[i]); t.faces[i] = prev; }
    }
    if (!get_positions(t.coverage_positions, h.coverage)) return fail("tile node reference out of range");
    t.coverage_class.assign(reinterpret_cast<const uint8_t*>(p), reinterpret_cast<const uint8_t*>(p) + h.coverage);
    for (int32_t f : t.faces) if (f < 0 || uint32_t(f) >= h.vertices) return fail("tile face index out of range");
    return true;
}

// ---------------------------------------------------------------------------------
// Tile build: gather the tile and its 8 neighbours, triangulate the tile only.
// ---------------------------------------------------------------------------------
struct BuildIo {
    const Quadtree* qt = nullptr;
    NodePoints* points = nullptr;
    Reader* rd = nullptr;
    uint64_t generation = 0;
    std::shared_mutex* io_lock = nullptr;
};
using KeptNodes = std::vector<std::pair<uint32_t, NodePoints::Ptr>>;

// The records of tile k: every record of its column nodes (they lie in the cell by
// construction) and the records of shallower leaves whose cell index is k.
inline bool gather_tile(const BuildIo& io, uint64_t k, SurfaceTile& out, KeptNodes& keep, uint64_t* shallow_in_cell, std::string* error = nullptr) {
    const Quadtree& qt = *io.qt;
    const auto& nodes = qt.nodes();
    const unsigned d = key_level(k);
    std::vector<uint32_t> col;
    if (const auto* c = qt.column(k)) col = *c;
    std::vector<uint32_t> sh;
    qt.shallow_leaves(k, sh);
    auto by_offset = [&](uint32_t a, uint32_t b) { return nodes[a].byte_offset < nodes[b].byte_offset; };
    std::sort(col.begin(), col.end(), by_offset);
    std::sort(sh.begin(), sh.end(), by_offset);
    double mn[2], mx[2];
    qt.cell_xy(k, mn, mx);
    out.cmin[0] = mn[0]; out.cmin[1] = mn[1]; out.cmax[0] = mx[0]; out.cmax[1] = mx[1];
    qt.z_range(k, out.cmin[2], out.cmax[2]);
    out.point_refs.clear();
    out.pts.clear();
    out.classes.reset();
    uint64_t in_cell = 0;
    for (uint32_t i : col) {
        auto v = io.points->get(i, *io.rd, nodes, io.generation, io.io_lock, error);
        if (!v) return false;
        for (const auto& p : *v) {
            if (io.rd->is_collection()) {
                uint32_t cx,cy;qt.record_cell(i,p.x,p.y,d,cx,cy);
                if(cx!=key_x(k)||cy!=key_y(k))continue;
            }
            out.point_refs.push_back(&p); out.classes.set(p.cls);
        }
        keep.emplace_back(i, std::move(v));
    }
    for (uint32_t i : sh) {
        auto v = io.points->get(i, *io.rd, nodes, io.generation, io.io_lock, error);
        if (!v) return false;
        for (const auto& p : *v) {
            uint32_t cx, cy;
            qt.record_cell(i, p.x, p.y, d, cx, cy);
            if (cx == key_x(k) && cy == key_y(k)) {
                out.point_refs.push_back(&p); out.classes.set(p.cls); ++in_cell;
            }
        }
        keep.emplace_back(i, std::move(v));
    }
    if (shallow_in_cell) *shallow_in_cell = in_cell;
    return true;
}

inline bool build_tile(const BuildIo& io, uint64_t k, const SurfaceParams& input, TileData& out, std::string* err) {
    SurfaceParams P = input;
    if (key_level(k) == 0) {
        // The root is also the continuous interpolation underlay. Keeping its
        // Delaunay faces fills sparse interiors after the cut refines.
        P.max_edge_abs = -1.0;
        P.min_area = P.min_aspect = 0.0;
    }
    const Quadtree& qt = *io.qt;
    const unsigned d = key_level(k);
    const int64_t n = int64_t(1) << d;
    const int64_t kx = key_x(k), ky = key_y(k);
    KeptNodes keep;
    std::vector<SurfaceTile> cells;
    std::vector<int64_t> cell_dx, cell_dy;
    cells.reserve(9);
    uint64_t shallow = 0;
    int self = -1;
    // Row-major (y, then x) order == ascending global index, so the mesher's "lowest
    // vector index owns" rule is the global "lowest tile index owns" rule.
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const int64_t x = kx + dx, y = ky + dy;
            if (x < 0 || y < 0 || x >= n || y >= n) continue;
            const uint64_t nk = make_key(d, uint32_t(x), uint32_t(y));
            if (nk != k && !qt.exists(nk)) continue;
            cells.emplace_back();
            if (!gather_tile(io, nk, cells.back(), keep, nk == k ? &shallow : nullptr, err)) {
                if (err && err->empty()) *err = "tile source read failed";
                return false;
            }
            if (nk == k) self = int(cells.size() - 1);
            cell_dx.push_back(dx); cell_dy.push_back(dy);
        }
    out = TileData{};
    out.key = k;
    out.flags = qt.exact(k) ? kFlagExact : 0;
    out.source_points = qt.leaf_points_inside(k) + shallow;
    if (io.rd->is_collection() && qt.exact(k)) out.source_points = cells[size_t(self)].point_count();
    std::vector<const SurfaceTile*> refs;
    for (const auto& c : cells) refs.push_back(&c);
    const detail::TileXIndex xindex(refs);
    detail::TileMesh part;
    part.collect_fringe = true;
    double L[3];
    surface_light_vector(P.azimuth_deg, P.elevation_deg, L);
    detail::mesh_one_tile(refs, size_t(self), xindex, P, L, part);
    // Source node of a record: the decoded vector whose storage holds it.
    std::vector<std::tuple<const SurfacePoint*, const SurfacePoint*, uint32_t>> spans;
    for (const auto& kv : keep)
        if (!kv.second->empty()) spans.emplace_back(kv.second->data(), kv.second->data() + kv.second->size(), kv.first);
    std::sort(spans.begin(), spans.end());
    spans.erase(std::unique(spans.begin(), spans.end()), spans.end());
    auto node_of = [&](const SurfacePoint* q) -> uint32_t {
        auto it = std::upper_bound(spans.begin(), spans.end(), q,
                                   [](const SurfacePoint* a, const auto& s) { return std::less<const SurfacePoint*>()(a, std::get<0>(s)); });
        if (it == spans.begin()) return UINT32_MAX;
        --it;
        return std::less<const SurfacePoint*>()(q, std::get<1>(*it)) ? std::get<2>(*it) : UINT32_MAX;
    };
    out.positions = std::move(part.positions);
    out.vertex_class = std::move(part.vertex_class);
    out.vertex_node.reserve(part.vertex_src.size());
    for (const SurfacePoint* q : part.vertex_src) out.vertex_node.push_back(node_of(q));
    out.faces = std::move(part.faces);
    out.groups[0] = uint32_t(out.faces.size() / 3);
    static const int kGroupDx[4] = {-1, -1, 0, 1}, kGroupDy[4] = {0, -1, -1, -1};
    for (int g = 0; g < 4; ++g) {
        int owner = -1;
        for (size_t c = 0; c < cells.size(); ++c)
            if (cell_dx[c] == kGroupDx[g] && cell_dy[c] == kGroupDy[g]) owner = int(c);
        if (owner < 0) continue;
        for (size_t f = 0; f < part.fringe_owner.size(); ++f)
            if (part.fringe_owner[f] == uint32_t(owner)) {
                out.faces.insert(out.faces.end(), part.fringe_faces.begin() + f * 3, part.fringe_faces.begin() + f * 3 + 3);
                ++out.groups[g + 1];
            }
    }
    const SurfaceTile& T = cells[size_t(self)];
    for (uint32_t i : part.coverage_point_indices) {
        const SurfacePoint& p = T.point(i);
        out.coverage_positions.insert(out.coverage_positions.end(), {p.x, p.y, p.z});
        out.coverage_class.push_back(p.cls);
        out.coverage_node.push_back(node_of(&p));
    }
    out.represented = uint32_t(part.visible_points);
    out.referenced = out.represented - out.coverage();
    for (int a = 0; a < 3; ++a) { out.bmin[a] = 1e300; out.bmax[a] = -1e300; }
    auto grow = [&](const std::vector<double>& v) {
        for (size_t i = 0; i < v.size(); ++i) {
            out.bmin[i % 3] = std::min(out.bmin[i % 3], v[i]);
            out.bmax[i % 3] = std::max(out.bmax[i % 3], v[i]);
        }
    };
    grow(out.positions);
    grow(out.coverage_positions);
    if (out.empty()) for (int a = 0; a < 3; ++a) out.bmin[a] = out.bmax[a] = 0.0;
    return true;
}

// The tiled path's geometry parameters: the compact crisp multi-class mesh with the
// whole-cut builder's filters, the absolute edge limit taken from the FILE's plan
// extent (not a camera cut), so a tile's triangles never depend on the view.
inline SurfaceParams lod_params(const SurfaceParams& in, const Quadtree& qt) {
    SurfaceParams P = in;
    P.method = SurfaceMethod::Facet;
    P.bake_colors = false;
    P.compact_crisp = true;
    P.color_source = SurfaceColor::Class;
    P.min_area = 1e-12;
    P.min_aspect = 1e-10;
    P.max_edge_abs = qt.xy_extent() * 0.10;
    if (const char* hf = std::getenv("NAKSHA_LNT_HALO_FRACTION")) {
        char* end = nullptr;
        const double f = std::strtod(hf, &end);
        if (end != hf && *end == '\0' && std::isfinite(f) && f >= 0.01 && f <= 0.5) P.halo_fraction = f;
    }
    return P;
}

// Geometry identity of a settings slot: anything that changes which triangles exist.
inline Digest settings_digest(const SurfaceParams& P) {
    embedded_surface::Hash h;
    const uint32_t algorithm = kAlgorithm;
    h.value(algorithm);
    h.add(P.class_visible, sizeof(P.class_visible));
    h.value(P.max_edge_factor); h.value(P.max_edge_abs); h.value(P.halo_fraction);
    h.value(P.min_area); h.value(P.min_aspect);
    return h.result();
}

// ---------------------------------------------------------------------------------
// Manager: workers, archive I/O, selection, GPU residency.
// ---------------------------------------------------------------------------------
enum LodStat : uint32_t {
    kStatDrawnTiles, kStatDrawnExactTiles, kStatSourceCovered, kStatSourceExact,
    kStatRepresentedDrawn, kStatTrianglesDrawn, kStatCoverageDrawn,
    kStatGpuTiles, kStatGpuVertices, kStatCpuTiles, kStatCpuBytes,
    kStatQueued, kStatInflight, kStatBuilt, kStatLoaded, kStatUploaded,
    kStatGpuEvicted, kStatCpuEvicted, kStatLoadFailures, kStatArchiveTiles,
    kStatCommits, kStatArchiveBytes, kStatTotalPoints, kStatWantedMissing,
    kStatCount
};

// Process environment as Windows holds it now (independent of which C runtime copy
// was initialised when), "" when unset.
inline std::string env_value(const char* name) {
    char buf[256];
    const DWORD n = GetEnvironmentVariableA(name, buf, sizeof(buf));
    return n > 0 && n < sizeof(buf) ? std::string(buf, n) : std::string();
}

struct LodConfig {
    double tau_px = 2.0;
    uint64_t gpu_vertices = 16u << 20, gpu_faces = 32u << 20, gpu_coverage = 4u << 20;
    uint64_t cpu_bytes = 768ull << 20;
    uint64_t node_bytes = 512ull << 20;
    uint64_t upload_bytes_per_frame = 32ull << 20;
    uint32_t uploads_per_frame = 24;
    unsigned workers = 0;
    bool persist = true;
    void from_env() {
        auto num = [](const char* name, double lo, double hi, double& v) {
            const std::string e = env_value(name);
            if (e.empty()) return;
            char* end = nullptr;
            const double x = std::strtod(e.c_str(), &end);
            if (end != e.c_str() && *end == '\0' && x >= lo && x <= hi) v = x;
        };
        num("NAKSHA_LNT_SURFACE_LOD_PX", 0.25, 64.0, tau_px);
        double gpu = 0.0, cpu = double(cpu_bytes >> 20);
        num("NAKSHA_LNT_SURFACE_GPU_MB", 64.0, 16384.0, gpu);
        if (gpu > 0.0) {                         // 16 B/vertex, 12 B/face, 16 B/coverage point
            const double v = gpu * 1048576.0 / 44.0;
            gpu_vertices = uint64_t(v); gpu_faces = uint64_t(2 * v); gpu_coverage = uint64_t(v / 4);
        }
        num("NAKSHA_LNT_SURFACE_CPU_MB", 64.0, 65536.0, cpu);
        cpu_bytes = uint64_t(cpu) << 20;
        if (env_value("NAKSHA_LNT_SURFACE_PERSIST") == "0") persist = false;
    }
};

class Manager {
public:
    Manager() = default;
    Manager(const Manager&) = delete;
    Manager& operator=(const Manager&) = delete;
    ~Manager() { shutdown(); }

    LodConfig config;

    // `io_lock` serialises payload reads against in-place class edits (may be null).
    bool open(const Reader& rd, const std::string& path, std::shared_mutex* io_lock, std::string* err) {
        path_ = path;
        io_lock_ = io_lock;
        index_offset_ = rd.get_index_offset();
        base_end_ = embedded_surface::base_end(rd);
        total_points_ = rd.directory().total_points;
        if (!qt_.build(rd)) { if (err) *err = "surface LOD: empty or degenerate directory"; return false; }
        points_ = std::make_unique<NodePoints>(config.node_bytes);
        points_->clear(source_gen_);
        if (rd.is_collection()) { config.persist = false; w_disabled_ = true; }
        else read_archive();
        const unsigned hw = std::max(2u, std::thread::hardware_concurrency());
        const unsigned n = config.workers ? config.workers : std::min(6u, std::max(2u, hw / 2));
        stop_ = false;
        for (unsigned i = 0; i < n; ++i) workers_.emplace_back(&Manager::worker_main, this);
        writer_ = std::thread(&Manager::writer_main, this);
        return true;
    }

    void shutdown() {
        verify_cancel_ = true;
        {
            std::lock_guard<std::mutex> lk(q_mu_);
            if (stop_ && workers_.empty() && !writer_.joinable()) return;
            stop_ = true;
            queue_.clear();
        }
        q_cv_.notify_all();
        for (auto& t : workers_) t.join();
        workers_.clear();
        {
            std::lock_guard<std::mutex> lk(w_mu_);
            w_stop_ = true;
        }
        w_cv_.notify_all();
        if (writer_.joinable()) writer_.join();
        if (verifier_.joinable()) { verify_cancel_ = true; verifier_.join(); }
    }

    const Quadtree& quadtree() const { return qt_; }

    // Geometry-affecting parameters (class visibility, edge limits). Returns true when
    // they select a different settings slot (tiles are reloaded / rebuilt for it).
    bool set_params(const SurfaceParams& in) {
        const SurfaceParams P = lod_params(in, qt_);
        const Digest d = settings_digest(P);
        if (have_params_ && d == settings_) return false;
        auto snap = std::make_shared<Snapshot>();
        snap->params = P;
        snap->settings = d;
        snap->params_gen = ++params_gen_;
        {
            std::lock_guard<std::mutex> lk(q_mu_);
            snapshot_ = snap;
            queue_.clear();
        }
        {
            std::lock_guard<std::mutex> lk(a_mu_);
            settings_ = d;
            have_params_ = true;
        }
        refresh_slot_entries();
        drop_tiles(true);
        return true;
    }

    // Class palette (256 x RGB) and the coverage-point shade. Colour only.
    void set_palette(const uint8_t* lut_rgb, float coverage_shade) {
        if (lut_rgb) std::memcpy(lut_, lut_rgb, sizeof(lut_));
        shade_ = coverage_shade;
        palette_dirty_ = true;
    }

    // The original records changed (in-place class edit, or another writer).
    void invalidate_source() {
        {
            std::lock_guard<std::mutex> lk(q_mu_);
            ++source_gen_;
            queue_.clear();
        }
        points_->clear(source_gen_);
        {
            std::lock_guard<std::mutex> lk(a_mu_);
            arch_ = surface_archive::State{};
            slot_entries_.clear();
            source_known_ = false;
            trusted_ = false;
        }
        {
            std::lock_guard<std::mutex> lk(w_mu_);
            pending_.clear();
            pending_bytes_ = 0;
        }
        drop_tiles(true);
    }

    // Class edits of stored records, shown without re-meshing: every resident tile vertex
    // or coverage point made from an edited record is recoloured in place (its CPU tile
    // copy-on-write, its GPU colours by a sparse upload), because a class change moves no
    // point and the compact mesh depends on classes only through their visibility. An
    // edit that hides or reveals a point (old/new class visibility differs) changes the
    // triangles: only the tiles around it are rebuilt. Work in flight is discarded and
    // the archive is re-keyed, as for any source change. `pending` = the edits are not in
    // the file yet: records read later get them too, until end_edit().
    struct PatchResult { uint32_t tiles = 0, vertices = 0, coverage = 0, rebuilt = 0; };
    PatchResult patch_classes(const std::vector<ClassEdit>& edits, const uint8_t* class_visible,
                              VulkanEngine* eng, bool pending) {
        PatchResult r;
        if (edits.empty()) return r;
        edits_pending_ = pending;
        uint64_t gen;
        {
            std::lock_guard<std::mutex> lk(q_mu_);
            gen = ++source_gen_;
            queue_.clear();
        }
        points_->apply_edits(gen, edits, pending);
        {
            std::lock_guard<std::mutex> lk(a_mu_);
            arch_ = surface_archive::State{};
            slot_entries_.clear();
            source_known_ = false;
            trusted_ = false;
        }
        struct PosKey {
            uint64_t a, b, c;
            bool operator==(const PosKey& o) const { return a == o.a && b == o.b && c == o.c; }
        };
        struct PosHash {
            size_t operator()(const PosKey& k) const {
                uint64_t h = k.a * 0x9E3779B97F4A7C15ull;
                h ^= k.b + 0x632BE59BD9B4E019ull + (h << 6) + (h >> 2);
                h ^= k.c + 0x85EBCA77C2B2AE63ull + (h << 6) + (h >> 2);
                return size_t(h);
            }
        };
        auto key_of = [](double x, double y, double z) {
            PosKey k;
            std::memcpy(&k.a, &x, 8); std::memcpy(&k.b, &y, 8); std::memcpy(&k.c, &z, 8);
            return k;
        };
        // Keep only edits that can affect a currently resident CPU surface.
        // A wide selection can contain millions of leaf/LOD records while the
        // rendered surface represents only a few hundred thousand vertices.
        // Hashing every edit made refresh cost scale with source-point count.
        std::unordered_set<PosKey, PosHash> resident_positions;
        size_t resident_count = 0;
        for (const auto& c : cpu_)
            resident_count += c.second.tile->vertex_class.size() + c.second.tile->coverage_class.size();
        const bool filter_resident = resident_count && edits.size() > resident_count * 2;
        if (filter_resident) {
            resident_positions.reserve(resident_count * 2);
            for (const auto& c : cpu_) {
                const TileData& t = *c.second.tile;
                for (size_t i = 0; i < t.vertex_class.size(); ++i)
                    resident_positions.insert(key_of(t.positions[i * 3], t.positions[i * 3 + 1], t.positions[i * 3 + 2]));
                for (size_t i = 0; i < t.coverage_class.size(); ++i)
                    resident_positions.insert(key_of(t.coverage_positions[i * 3], t.coverage_positions[i * 3 + 1],
                                                     t.coverage_positions[i * 3 + 2]));
            }
        }
        std::unordered_map<PosKey, uint8_t, PosHash> recolour;
        std::vector<std::pair<double, double>> reshape;           // visibility changes
        double lo[2] = {1e300, 1e300}, hi[2] = {-1e300, -1e300};
        bool recolour_requested = false;
        recolour.reserve((filter_resident ? std::min(edits.size(), resident_count) : edits.size()) * 2);
        for (const auto& e : edits) {
            const bool was = class_visible[e.old_cls] != 0, now = class_visible[e.cls] != 0;
            if (was != now) { reshape.push_back({e.x, e.y}); continue; }
            if (!now) continue;                                   // hidden before and after
            recolour_requested = true;
            lo[0] = std::min(lo[0], e.x); lo[1] = std::min(lo[1], e.y);
            hi[0] = std::max(hi[0], e.x); hi[1] = std::max(hi[1], e.y);
            const PosKey pos = key_of(e.x, e.y, e.z);
            if (!filter_resident || resident_positions.find(pos) != resident_positions.end()) recolour[pos] = e.cls;
        }
        // A tile's vertices lie in its cell or (halo / fringe) its eight neighbours.
        auto reach = [&](uint64_t k, double mn[2], double mx[2]) {
            qt_.cell_xy(k, mn, mx);
            for (int a = 0; a < 2; ++a) { const double w = mx[a] - mn[a]; mn[a] -= w; mx[a] += w; }
        };
        std::unordered_set<uint64_t> keys;
        for (const auto& g : gpu_) keys.insert(g.first);
        for (const auto& c : cpu_) keys.insert(c.first);
        std::vector<uint64_t> rebuild;
        std::vector<uint32_t> vi, ci;
        std::vector<uint8_t> vc, cc;
        for (uint64_t k : keys) {
            double mn[2], mx[2];
            reach(k, mn, mx);
            bool redo = false;
            for (const auto& p : reshape)
                if (p.first >= mn[0] && p.first <= mx[0] && p.second >= mn[1] && p.second <= mx[1]) { redo = true; break; }
            if (!redo && (!recolour_requested || hi[0] < mn[0] || lo[0] > mx[0] || hi[1] < mn[1] || lo[1] > mx[1]))
                continue;
            auto cit = cpu_.find(k);
            if (!redo && cit == cpu_.end()) redo = true;          // no copy to find the vertices in
            if (redo) { rebuild.push_back(k); continue; }
            const TileData& t = *cit->second.tile;
            vi.clear(); vc.clear(); ci.clear(); cc.clear();
            auto scan = [&](const std::vector<double>& pos, const std::vector<uint8_t>& cls,
                            std::vector<uint32_t>& idx, std::vector<uint8_t>& out) {
                for (size_t i = 0; i < cls.size(); ++i) {
                    const double x = pos[i * 3], y = pos[i * 3 + 1];
                    if (x < lo[0] || x > hi[0] || y < lo[1] || y > hi[1]) continue;
                    auto it = recolour.find(key_of(x, y, pos[i * 3 + 2]));
                    if (it != recolour.end() && it->second != cls[i]) { idx.push_back(uint32_t(i)); out.push_back(it->second); }
                }
            };
            scan(t.positions, t.vertex_class, vi, vc);
            scan(t.coverage_positions, t.coverage_class, ci, cc);
            if (vi.empty() && ci.empty()) continue;
            auto copy = cit->second.tile.use_count() == 1
                ? std::const_pointer_cast<TileData>(cit->second.tile)
                : std::make_shared<TileData>(t);
            for (size_t i = 0; i < vi.size(); ++i) copy->vertex_class[vi[i]] = vc[i];
            for (size_t i = 0; i < ci.size(); ++i) copy->coverage_class[ci[i]] = cc[i];
            cit->second.tile = copy;
            auto git = gpu_.find(k);
            if (git != gpu_.end() && !git->second.empty) {
                if (!eng || !eng->surface_tiles_patch_classes(k, vi, vc, ci, cc)) { rebuild.push_back(k); continue; }
            }
            ++r.tiles;
            r.vertices += uint32_t(vi.size());
            r.coverage += uint32_t(ci.size());
        }
        if (!rebuild.empty()) {
            std::vector<uint64_t> evict;
            for (uint64_t k : rebuild) {
                auto git = gpu_.find(k);
                if (git != gpu_.end()) {
                    if (!git->second.empty) {
                        evict.push_back(k);
                        gpu_vertices_ -= git->second.vertices; gpu_faces_ -= git->second.faces;
                        gpu_coverage_ -= git->second.coverage;
                    }
                    gpu_.erase(git);
                    gpu_bounds_.erase(k);
                }
                auto cit = cpu_.find(k);
                if (cit != cpu_.end()) { cpu_bytes_ -= cit->second.tile->bytes(); cpu_.erase(cit); }
                failed_.erase(k);
            }
            if (eng && !evict.empty()) eng->surface_tiles_evict(evict);
            last_draw_.clear();
            r.rebuilt = uint32_t(rebuild.size());
        }
        // The archive is re-keyed by the edit: the resident tiles (patched or untouched,
        // all current) are re-encoded and committed by the writer, off this thread, so a
        // reopen loads them instead of triangulating again.
        {
            std::lock_guard<std::mutex> lk(w_mu_);
            pending_.clear();
            pending_bytes_ = 0;
            if (!pending && config.persist && !w_disabled_) {
                if (reencode_.empty()) pending_since_ = std::chrono::steady_clock::now();
                for (const auto& c : cpu_) reencode_[c.first] = {c.second.tile, gen};
            }
        }
        w_cv_.notify_all();
        return r;
    }
    // The edited records are in the file now: later reads see them without help.
    void end_edit() {
        points_->clear_overrides();
        edits_pending_ = false;
        std::lock_guard<std::mutex> lk(w_mu_);
        if (config.persist && !w_disabled_) {
            if (reencode_.empty()) pending_since_ = std::chrono::steady_clock::now();
            for (const auto& c : cpu_) reencode_[c.first] = {c.second.tile, source_gen_.load()};
            w_cv_.notify_all();
        }
    }

    // Leaving the LOD path (another surface took the engine): GPU residency is gone.
    void deactivate(VulkanEngine& eng) {
        if (gpu_configured_) eng.surface_tiles_release();
        gpu_configured_ = false;
        gpu_.clear();
        gpu_vertices_ = gpu_faces_ = gpu_coverage_ = 0;
        last_draw_.clear();
        drawn_.clear();
    }

    // Once per frame on the render thread. Returns true when what is drawn changed.
    bool update(const LNTCamera& cam, VulkanEngine& eng, const BoundingBox* corridor, std::string* err) {
        ++frame_;
        if (!have_params_) return false;
        if (external_change_.exchange(false)) {
            std::fprintf(stderr, "[LNT surface] the .lnt file changed on disk; cached surface tiles discarded\n");
            invalidate_source();
        }
        if (!gpu_configured_ || !eng.surface_tiles_active()) {
            if (gpu_configured_) { gpu_.clear(); gpu_vertices_ = gpu_faces_ = gpu_coverage_ = 0; last_draw_.clear(); }
            if (!configure_gpu(eng, err)) return false;
        }
        if (palette_dirty_) {
            eng.surface_tiles_set_palette(lut_, shade_);
            palette_dirty_ = false;
        }
        drain_completions();
        select(cam, corridor);
        upload(eng);
        evict_cpu();
        // Requests for the worker pool (missing tiles, priority order).
        {
            std::vector<Job> jobs;
            jobs.reserve(wanted_.size());
            for (const auto& w : wanted_)
                if (!cpu_.count(w.second) && !gpu_.count(w.second) && !failed_.count(w.second))
                    jobs.push_back({w.second, w.first});
            std::lock_guard<std::mutex> lk(q_mu_);
            // A job may have finished after this frame drained completions: its result
            // is waiting in completions_ and must not be queued (built) a second time.
            std::unordered_set<uint64_t> finished;
            for (const auto& c : completions_) finished.insert(c.key);
            queue_.clear();
            for (const auto& j : jobs) if (!inflight_.count(j.key) && !finished.count(j.key)) queue_.push_back(j);
            stats_[kStatQueued] = queue_.size();
            stats_[kStatInflight] = inflight_.size();
        }
        q_cv_.notify_all();
        // Draw list: owned triangles always; a fringe group where its owner is not drawn.
        std::vector<VulkanEngine::SurfaceTileDraw> items;
        std::unordered_set<uint64_t> drawn(drawn_.begin(), drawn_.end());
        uint64_t s_cov = 0, s_exact = 0, rep = 0, tris = 0, cov = 0, exact_tiles = 0;
        for (uint64_t k : drawn_) {
            auto it = gpu_.find(k);
            if (it == gpu_.end()) continue;
            GpuTile& g = it->second;
            g.last_frame = frame_;
            s_cov += g.source_points;
            rep += g.represented;
            if (g.exact) { s_exact += g.source_points; ++exact_tiles; }
            if (g.empty) continue;
            const unsigned d = key_level(k);
            const int64_t x = key_x(k), y = key_y(k);
            static const int kDx[4] = {-1, -1, 0, 1}, kDy[4] = {0, -1, -1, -1};
            uint32_t first = 0;
            for (int grp = 0; grp < 5; ++grp) {
                const uint32_t count = g.groups[grp];
                bool show = grp == 0;
                if (grp > 0 && count) {
                    const int64_t nx = x + kDx[grp - 1], ny = y + kDy[grp - 1];
                    show = nx < 0 || ny < 0 || !drawn.count(make_key(d, uint32_t(nx), uint32_t(ny)));
                }
                if (show && (count || (grp == 0 && g.coverage))) {
                    items.push_back({k, first, count, grp == 0 ? 1u : 0u, 0u});
                    tris += count;
                    if (grp == 0) cov += g.coverage;
                }
                first += count;
            }
        }
        const uint64_t root = make_key(0, 0, 0);
        auto background = gpu_.find(root);
        if (!drawn.count(root) && background != gpu_.end()) {
            auto& g = background->second;
            g.last_frame = frame_;
            if (!g.empty && g.groups[0]) {
                items.push_back({root, 0, g.groups[0], 0, 1});
                tris += g.groups[0];
            }
        }
        const bool changed = items.size() != last_draw_.size() ||
            !std::equal(items.begin(), items.end(), last_draw_.begin(), [](const auto& a, const auto& b) {
                return a.key == b.key && a.first_face == b.first_face && a.face_count == b.face_count && a.coverage == b.coverage && a._pad == b._pad;
            });
        if (changed) {
            eng.surface_tiles_set_draw_list(items);
            last_draw_ = items;
        }
        stats_[kStatDrawnTiles] = drawn_.size();
        stats_[kStatDrawnExactTiles] = exact_tiles;
        stats_[kStatSourceCovered] = s_cov;
        stats_[kStatSourceExact] = s_exact;
        stats_[kStatRepresentedDrawn] = rep;
        stats_[kStatTrianglesDrawn] = tris;
        stats_[kStatCoverageDrawn] = cov;
        stats_[kStatGpuTiles] = gpu_.size();
        stats_[kStatGpuVertices] = gpu_vertices_;
        stats_[kStatCpuTiles] = cpu_.size();
        stats_[kStatCpuBytes] = cpu_bytes_;
        stats_[kStatTotalPoints] = total_points_;
        stats_[kStatWantedMissing] = wanted_missing_;
        stats_[kStatLoadFailures] = load_failures_.load();
        stats_[kStatCommits] = commits_.load();
        stats_[kStatArchiveBytes] = archive_bytes_.load();
        {
            std::lock_guard<std::mutex> lk(a_mu_);
            stats_[kStatArchiveTiles] = slot_entries_.size();
        }
        return changed;
    }

    // True while the wanted cut is not yet fully resident. Archive verification
    // is background maintenance and must not keep a rendered viewport busy.
    bool busy() const {
        std::lock_guard<std::mutex> lk(q_mu_);
        return wanted_missing_ != 0 || !queue_.empty() || !inflight_.empty() || !completions_.empty();
    }
    bool verifying() const { return verifying_.load(); }
    // First tile that could not be built or loaded (its area shows a coarser tile), or "".
    const std::string& failure() const { return failure_; }
    const uint64_t* stats() const { return stats_; }
    // Latest persistence outcome worth reporting once (empty when nothing new).
    std::string take_persist_message() {
        std::lock_guard<std::mutex> lk(w_mu_);
        std::string s;
        s.swap(persist_message_);
        return s;
    }
    // Blocks until every built tile has been committed (or persistence gave up).
    void flush() {
        std::unique_lock<std::mutex> lk(w_mu_);
        w_flush_ = true;
        w_cv_.notify_all();
        w_done_cv_.wait_for(lk, std::chrono::seconds(60), [&] {
            return (pending_.empty() && reencode_.empty() && !writer_busy_) || w_disabled_ || w_stop_; });
        w_flush_ = false;
    }

private:
    struct Snapshot { SurfaceParams params; Digest settings{}; uint64_t params_gen = 0; };
    struct Info { bool exact = false; double est = 0.0, z0 = 0.0, z1 = 0.0; std::vector<uint64_t> kids; };
    struct Job { uint64_t key; double priority; };
    struct Completion {
        uint64_t key = 0, source_gen = 0, params_gen = 0;
        std::shared_ptr<const TileData> tile;
        bool loaded = false;
        std::string error;
    };
    struct CpuTile { std::shared_ptr<const TileData> tile; uint64_t last_frame = 0; };
    struct GpuTile {
        uint32_t groups[5]{};
        uint32_t vertices = 0, faces = 0, coverage = 0, represented = 0;
        uint64_t source_points = 0, last_frame = 0;
        bool exact = false, empty = false;
    };

    // ---- archive read side ------------------------------------------------
    void read_archive() {
        surface_archive::File f;
        std::string e;
        if (!f.open_read(path_, &e)) { std::fprintf(stderr, "[LNT surface] %s\n", e.c_str()); return; }
        Digest dir{};
        if (!surface_archive::directory_digest(f, index_offset_, base_end_, dir)) return;
        dir_hash_ = dir;
        surface_archive::State st = surface_archive::read_state(f, base_end_, dir);
        if (!st.present) {
            if (st.legacy_tail) std::fprintf(stderr, "[LNT surface] legacy whole-file shading mesh found; the tiled archive replaces it on its first commit\n");
            return;
        }
        if (st.stamp_ok()) {
            std::lock_guard<std::mutex> lk(a_mu_);
            arch_ = st;
            source_hash_ = st.footer.source_hash;
            source_mtime_ = st.mtime;
            source_known_ = trusted_ = true;
            std::fprintf(stderr, "[LNT surface] tiled surface archive: %zu tiles, commit stamp valid (original data unchanged)\n", st.entries.size());
            return;
        }
        // Stamp differs: someone else wrote the file. Re-hash the original bytes before
        // any stored tile is trusted; meanwhile nothing is loaded from the archive.
        verifying_ = true;
        verify_cancel_ = false;
        verifier_ = std::thread([this, st]() mutable {
            surface_archive::File vf;
            std::string ve;
            Digest h{};
            int64_t m = 0;
            bool ok = vf.open_read(path_, &ve);
            if (ok) { m = vf.mtime(); ok = surface_archive::source_digest(vf, base_end_, h, &verify_cancel_); }
            {
                std::lock_guard<std::mutex> lk(a_mu_);
                if (ok && h == st.footer.source_hash) {
                    arch_ = st;
                    trusted_ = true;
                    std::fprintf(stderr, "[LNT surface] tiled surface archive re-verified against the original data (%zu tiles)\n", st.entries.size());
                } else if (ok) {
                    std::fprintf(stderr, "[LNT surface] original .lnt data changed since the surface archive was written; it will be rebuilt\n");
                }
                if (ok) { source_hash_ = h; source_mtime_ = m; source_known_ = true; }
            }
            refresh_slot_entries();
            verifying_ = false;
            q_cv_.notify_all();
        });
    }

    void refresh_slot_entries() {
        std::lock_guard<std::mutex> lk(a_mu_);
        slot_entries_.clear();
        if (!trusted_ || !arch_.present || !have_params_) return;
        const int slot = arch_.slot_of(settings_);
        if (slot < 0) return;
        for (const auto& e : arch_.entries) if (e.slot == slot) slot_entries_[e.key] = e;
    }

    // ---- workers ----------------------------------------------------------
    void worker_main() {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
        Reader rd;
        bool rd_open = false;
        surface_archive::File af;
        bool af_open = false;
        embedded_surface::MeshCodec codec;
        for (;;) {
            Job job{};
            std::shared_ptr<const Snapshot> snap;
            uint64_t sgen = 0;
            {
                std::unique_lock<std::mutex> lk(q_mu_);
                // Verification only gates reuse of stored geometry. Fresh
                // meshes read the current source and must remain available
                // while a multi-gigabyte source fingerprint runs in the
                // background; otherwise even the root stays blank for minutes.
                q_cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
                if (stop_) return;
                auto best = std::min_element(queue_.begin(), queue_.end(), [](const Job& a, const Job& b) { return a.priority < b.priority; });
                job = *best;
                queue_.erase(best);
                inflight_.insert(job.key);
                snap = snapshot_;
                sgen = source_gen_;
            }
            Completion c;
            c.key = job.key; c.source_gen = sgen; c.params_gen = snap ? snap->params_gen : 0;
            try {
                auto tile = std::make_shared<TileData>();
                bool done = false;
                surface_archive::IndexEntry entry{};
                bool has_entry = false;
                {
                    std::lock_guard<std::mutex> lk(a_mu_);
                    auto it = slot_entries_.find(job.key);
                    if (it != slot_entries_.end()) { entry = it->second; has_entry = true; }
                }
                if (has_entry && snap) {
                    if (!af_open) af_open = af.open_read(path_, nullptr);
                    std::vector<char> raw;
                    std::string why;
                    if (af_open && surface_archive::read_record(af, entry, snap->settings, raw, codec, &why) &&
                        decode_tile(raw, job.key, qt_, *tile, &why)) {
                        done = true;
                        c.loaded = true;
                    } else {
                        std::fprintf(stderr, "[LNT surface] stored tile %016llx rejected (%s); rebuilding it\n",
                                     (unsigned long long)job.key, why.c_str());
                        std::lock_guard<std::mutex> lk(a_mu_);
                        slot_entries_.erase(job.key);
                        ++load_failures_;
                    }
                }
                if (!done && snap) {
                    if (!rd_open) rd_open = rd.open(path_);
                    if (!rd_open) throw std::runtime_error("surface LOD reader open failed: " + rd.error());
                    BuildIo io{&qt_, points_.get(), &rd, sgen, io_lock_};
                    std::string why;
                    if (!build_tile(io, job.key, snap->params, *tile, &why)) throw std::runtime_error(why);
                    // Do not race the untrusted archive verifier with a second
                    // full source hash in the writer. These fresh tiles remain
                    // in the normal session CPU/GPU caches during verification.
                    if (config.persist && !verifying_.load()) {
                        surface_archive::PendingRecord rec;
                        encode_tile(*tile, snap->settings, qt_, codec, rec);
                        std::lock_guard<std::mutex> lk(w_mu_);
                        if (sgen == source_gen_.load() && !w_disabled_ && !edits_pending_) {
                            pending_bytes_ += rec.stored.size();
                            if (pending_.empty()) pending_since_ = std::chrono::steady_clock::now();
                            pending_.push_back({std::move(rec), sgen});
                            w_cv_.notify_all();
                        }
                    }
                    std::vector<uint32_t>().swap(tile->vertex_node);
                    std::vector<uint32_t>().swap(tile->coverage_node);
                    done = true;
                }
                if (done) c.tile = std::move(tile);
            } catch (const std::exception& ex) {
                c.error = ex.what();
            }
            {
                std::lock_guard<std::mutex> lk(q_mu_);
                inflight_.erase(job.key);
                completions_.push_back(std::move(c));
            }
        }
    }

    // ---- writer: batches built tiles into archive commits --------------------
    struct PendingItem { surface_archive::PendingRecord rec; uint64_t source_gen; };
    // Resident tiles to re-encode and commit after a class edit (w_mu_), latest per key.
    std::unordered_map<uint64_t, std::pair<std::shared_ptr<const TileData>, uint64_t>> reencode_;
    bool writer_busy_ = false; // guarded by w_mu_; flush includes encoding/commit work
    std::atomic<bool> edits_pending_{false}; // previews never persist uncommitted colours
    void writer_main() {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
        auto retry_at = std::chrono::steady_clock::time_point{};
        embedded_surface::MeshCodec codec;
        std::unordered_map<uint64_t, std::pair<std::shared_ptr<const TileData>, uint64_t>> reencode;
        for (;;) {
            std::vector<PendingItem> batch;
            {
                std::unique_lock<std::mutex> lk(w_mu_);
                w_cv_.wait_for(lk, std::chrono::milliseconds(500), [&] {
                    return w_stop_ || w_flush_ || pending_bytes_ >= (16ull << 20);
                });
                const auto now = std::chrono::steady_clock::now();
                const bool due = (!pending_.empty() || !reencode_.empty()) &&
                    (w_stop_ || w_flush_ || pending_bytes_ >= (16ull << 20) || now - pending_since_ >= std::chrono::seconds(2));
                if (w_disabled_) { pending_.clear(); reencode_.clear(); pending_bytes_ = 0; w_done_cv_.notify_all(); }
                if (!due || (now < retry_at && !w_stop_ && !w_flush_)) {
                    if (w_stop_) { w_done_cv_.notify_all(); return; }
                    continue;
                }
                batch.swap(pending_);
                pending_bytes_ = 0;
                reencode.swap(reencode_);
                writer_busy_ = true;
            }
            if (!reencode.empty()) {
                Digest settings;
                {
                    std::lock_guard<std::mutex> lk(a_mu_);
                    settings = settings_;
                }
                for (const auto& kv : reencode) {
                    if (kv.second.second != source_gen_.load()) continue;     // edited again
                    surface_archive::PendingRecord rec;
                    if (encode_tile(*kv.second.first, settings, qt_, codec, rec))
                        batch.push_back({std::move(rec), kv.second.second});
                }
                reencode.clear();
            }
            const bool ok = commit_batch(batch);
            std::lock_guard<std::mutex> lk(w_mu_);
            writer_busy_ = false;
            if (!ok && !w_disabled_ && !batch.empty()) {
                // Transient failure (lock held elsewhere): keep the tiles, retry later.
                for (auto& b : batch) { pending_bytes_ += b.rec.stored.size(); pending_.push_back(std::move(b)); }
                if (pending_since_ == std::chrono::steady_clock::time_point{}) pending_since_ = std::chrono::steady_clock::now();
                retry_at = std::chrono::steady_clock::now() + std::chrono::seconds(15);
                if (w_stop_ || w_flush_) { pending_.clear(); pending_bytes_ = 0; }
                if (w_stop_) { w_done_cv_.notify_all(); return; }
            }
            w_done_cv_.notify_all();
            if (w_stop_ && pending_.empty()) return;
        }
    }

    bool commit_batch(std::vector<PendingItem>& batch) {
        if (edits_pending_) { batch.clear(); return true; }
        const uint64_t gen = source_gen_.load();
        std::vector<surface_archive::PendingRecord> recs;
        for (auto& b : batch) if (b.source_gen == gen) recs.push_back(std::move(b.rec));
        batch.clear();
        if (recs.empty()) return true;
        Digest src{};
        int64_t mtime = 0;
        bool known;
        {
            std::lock_guard<std::mutex> lk(a_mu_);
            known = source_known_;
            src = source_hash_;
            mtime = source_mtime_;
        }
        if (!known) {
            // First commit for this source: fingerprint the original bytes (the last-write
            // time is taken first, so a write during hashing is detected at commit).
            surface_archive::File f;
            std::string e;
            if (!f.open_read(path_, &e)) { report_persist(e, true); return false; }
            mtime = f.mtime();
            std::atomic<bool> never{false};
            if (!surface_archive::source_digest(f, base_end_, src, &never)) { report_persist("cannot fingerprint the original .lnt data", false); return false; }
            std::lock_guard<std::mutex> lk(a_mu_);
            if (gen != source_gen_.load()) return true;           // edited meanwhile: drop
            source_hash_ = src; source_mtime_ = mtime; source_known_ = true;
        }
        uint64_t bytes = 0;
        for (const auto& r : recs) bytes += r.stored.size();
        const auto t0 = std::chrono::steady_clock::now();
        // Growth bound: four times the original data (two complete class-visibility
        // configurations fit; ~20 B per record each), at least 256 MB.
        const uint64_t bound = std::max<uint64_t>(base_end_ * 4, 256ull << 20);
        surface_archive::CommitResult r = surface_archive::commit(path_, base_end_, index_offset_, src, mtime, recs, bound);
        if (!r.ok) {
            if (r.source_changed) {
                std::lock_guard<std::mutex> lk(a_mu_);
                source_known_ = false;
                external_change_ = true;
            }
            report_persist(r.error, r.permanent);
            return r.source_changed;      // changed source: those tiles are stale anyway
        }
        {
            std::lock_guard<std::mutex> lk(a_mu_);
            if (gen == source_gen_.load()) {
                arch_ = r.state;
                trusted_ = true;
                source_mtime_ = r.mtime;
            }
        }
        refresh_slot_entries();
        ++commits_;
        archive_bytes_ += bytes;
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "[LNT surface] archive commit: %zu tiles, %.1f MB in %.0f ms (%llu tiles stored; original point records unchanged)\n",
                     recs.size(), double(bytes) / 1e6, ms, (unsigned long long)r.state.entries.size());
        return true;
    }

    void report_persist(const std::string& error, bool permanent) {
        std::lock_guard<std::mutex> lk(w_mu_);
        const std::string msg = std::string("surface archive not saved: ") + error +
                                (permanent ? " (tiles stay cached for this session only)" : " (will retry)");
        if (msg != last_persist_message_) {
            std::fprintf(stderr, "[LNT surface] %s; original point records unchanged\n", msg.c_str());
            persist_message_ = msg;
            last_persist_message_ = msg;
        }
        if (permanent) { w_disabled_ = true; pending_.clear(); pending_bytes_ = 0; }
    }

    // ---- render thread ----------------------------------------------------
    bool configure_gpu(VulkanEngine& eng, std::string* err) {
        uint64_t v = config.gpu_vertices, f = config.gpu_faces, c = config.gpu_coverage;
        for (int attempt = 0; attempt < 3; ++attempt) {
            if (eng.surface_tiles_configure(v, f, c)) {
                gpu_caps_[0] = v; gpu_caps_[1] = f; gpu_caps_[2] = c;
                gpu_configured_ = true;
                palette_dirty_ = true;
                eng.surface_tiles_set_palette(lut_, shade_);
                palette_dirty_ = false;
                return true;
            }
            v /= 2; f /= 2; c /= 2;
        }
        if (err) *err = std::string("surface tile arena unavailable: ") + eng.last_error();
        return false;
    }

    void drop_tiles(bool cpu_too) {
        if (cpu_too) { cpu_.clear(); cpu_bytes_ = 0; }
        evict_all_gpu_ = true;
        failed_.clear();
        failure_.clear();
        refined_.clear();
    }

    void drain_completions() {
        std::vector<Completion> done;
        {
            std::lock_guard<std::mutex> lk(q_mu_);
            done.swap(completions_);
        }
        const uint64_t pgen = snapshot_ ? snapshot_->params_gen : 0;
        for (auto& c : done) {
            if (c.source_gen != source_gen_.load() || c.params_gen != pgen) continue;
            if (!c.tile) {
                if (!failed_.count(c.key)) {
                    std::fprintf(stderr, "[LNT surface] tile %016llx failed: %s\n", (unsigned long long)c.key, c.error.c_str());
                    if (failure_.empty()) {
                        char where[96];
                        std::snprintf(where, sizeof(where), "surface tile %u/%u/%u: ", key_level(c.key), key_x(c.key), key_y(c.key));
                        failure_ = where + c.error;
                    }
                }
                failed_.insert(c.key);
                continue;
            }
            ++stats_[c.loaded ? kStatLoaded : kStatBuilt];
            cpu_bytes_ += c.tile->bytes();
            cpu_[c.key] = {std::move(c.tile), frame_};
        }
    }

    // Draw cover + requests (see header). Recursion depth <= 27.
    bool cover(uint64_t k, const LNTCamera& cam, const BoundingBox* corridor, std::unordered_set<uint64_t>& refined_now) {
        double mn[2], mx[2], z0, z1;
        qt_.cell_xy(k, mn, mx);
        const double margin = 0.1 * (mx[0] - mn[0]);
        const Info& in = info(k);
        auto bit = gpu_bounds_.find(k);
        if (bit != gpu_bounds_.end()) { z0 = bit->second.first; z1 = bit->second.second; }
        else { z0 = in.z0; z1 = in.z1; }
        const BoundingBox box = BoundingBox::from_min_max(mn[0] - margin, mn[1] - margin, z0, mx[0] + margin, mx[1] + margin, z1);
        if (!cam.visible(box)) return true;
        if (corridor && !box.intersects(*corridor)) return true;
        const unsigned d = key_level(k);
        const bool resident = gpu_.count(k) != 0;
        if (resident) cover_keep_.insert(k);
        // Refine from a resident cover, one level at a time. Walking straight
        // to the leaves on a cold billion-point view floods the queue with
        // exact meshes before a usable coarse viewport has been uploaded.
        if (!resident && !failed_.count(k)) {
            const double cx = 0.5 * (mn[0] + mx[0]) - cam.target[0];
            const double cy = 0.5 * (mn[1] + mx[1]) - cam.target[1];
            wanted_.push_back({double(d) * 1e12 + cx * cx + cy * cy, k});
            return false;
        }
        const bool exact = in.exact;
        bool want = false;
        if (!exact) {
            const double side = mx[0] - mn[0];
            const double spacing = side / std::sqrt(std::max(1.0, in.est));
            double ppm;
            if (cam.ortho) ppm = cam.pixels_per_meter_at(1.0);
            else {
                double dist2 = 0.0;
                for (int a = 0; a < 3; ++a) {
                    const double lo = box.min[a], hi = box.max[a], e = cam.eye[a];
                    const double q = e < lo ? lo - e : (e > hi ? e - hi : 0.0);
                    dist2 += q * q;
                }
                ppm = cam.pixels_per_meter_at(std::max(std::sqrt(dist2), cam.near_clip));
            }
            const double px = spacing * ppm;
            want = px > config.tau_px || (refined_.count(k) && px > 0.7 * config.tau_px);
        }
        if (want) {
            const size_t mark = drawn_.size();
            bool ok = !in.kids.empty();
            for (uint64_t c : in.kids) ok = cover(c, cam, corridor, refined_now) && ok;
            if (ok) { refined_now.insert(k); return true; }
            drawn_.resize(mark);
        }
        if (gpu_.count(k)) { drawn_.push_back(k); return true; }
        if (!failed_.count(k)) {
            const double cx = 0.5 * (mn[0] + mx[0]) - cam.target[0], cy = 0.5 * (mn[1] + mx[1]) - cam.target[1];
            wanted_.push_back({double(d) * 1e12 + cx * cx + cy * cy, k});
        }
        return false;
    }

    // Static per-tile facts (directory only), cached on the render thread. Element
    // references stay valid across rehashing, so recursion may hold one.
    const Info& info(uint64_t k) {
        auto it = info_.find(k);
        if (it != info_.end()) return it->second;
        Info i;
        i.exact = qt_.exact(k);
        i.est = qt_.estimated_points(k);
        qt_.z_range(k, i.z0, i.z1);
        qt_.children(k, i.kids);
        return info_.emplace(k, std::move(i)).first->second;
    }

    void select(const LNTCamera& cam, const BoundingBox* corridor) {
        drawn_.clear();
        wanted_.clear();
        cover_keep_.clear();
        std::unordered_set<uint64_t> refined_now;
        cover(make_key(0, 0, 0), cam, corridor, refined_now);
        refined_.swap(refined_now);
        // Keep the coarse continuous TIN resident under the detailed cut. It
        // is drawn at background depth and cannot cover a detailed fragment.
        const uint64_t root = make_key(0, 0, 0);
        if (!gpu_.count(root) && !failed_.count(root)) {
            auto request = std::find_if(wanted_.begin(), wanted_.end(),
                [root](const auto& w) { return w.second == root; });
            if (request == wanted_.end()) wanted_.push_back({-1.0, root});
            else request->first = -1.0;
        }
        std::sort(wanted_.begin(), wanted_.end());
        wanted_missing_ = wanted_.size();
    }

    void upload(VulkanEngine& eng) {
        if (evict_all_gpu_) {
            std::vector<uint64_t> keys;
            for (const auto& g : gpu_) if (!g.second.empty) keys.push_back(g.first);
            if (!keys.empty()) eng.surface_tiles_evict(keys);
            stats_[kStatGpuEvicted] += keys.size();
            gpu_.clear(); gpu_bounds_.clear();
            gpu_vertices_ = gpu_faces_ = gpu_coverage_ = 0;
            last_draw_.clear();
            evict_all_gpu_ = false;
        }
        std::vector<std::shared_ptr<const TileData>> batch;
        uint64_t bytes = 0, need[3] = {0, 0, 0};
        for (const auto& w : wanted_) {
            if (batch.size() >= config.uploads_per_frame || bytes >= config.upload_bytes_per_frame) break;
            auto it = cpu_.find(w.second);
            if (it == cpu_.end() || gpu_.count(w.second)) continue;
            it->second.last_frame = frame_;
            const TileData& t = *it->second.tile;
            if (t.empty()) {
                GpuTile g;
                g.exact = t.exact(); g.empty = true; g.source_points = t.source_points; g.represented = t.represented;
                g.last_frame = frame_;
                gpu_[t.key] = g;
                continue;
            }
            batch.push_back(it->second.tile);
            bytes += uint64_t(t.vertices()) * 16 + uint64_t(t.triangles()) * 12 + uint64_t(t.coverage()) * 16;
            need[0] += t.vertices(); need[1] += t.triangles(); need[2] += t.coverage();
        }
        if (batch.empty()) return;
        make_room(eng, need);
        std::vector<VulkanEngine::SurfaceTileUpload> descs(batch.size());
        for (size_t i = 0; i < batch.size(); ++i) {
            const TileData& t = *batch[i];
            auto& d = descs[i];
            d.key = t.key;
            d.positions = t.positions.data(); d.vertex_class = t.vertex_class.data(); d.faces = t.faces.data();
            d.coverage_positions = t.coverage_positions.empty() ? nullptr : t.coverage_positions.data();
            d.coverage_class = t.coverage_class.empty() ? nullptr : t.coverage_class.data();
            d.vertex_count = t.vertices(); d.face_count = t.triangles(); d.coverage_count = t.coverage();
            d.result = 0;
        }
        eng.surface_tiles_upload(descs);
        bool full = false;
        for (size_t i = 0; i < batch.size(); ++i) {
            const TileData& t = *batch[i];
            if (descs[i].result == 1) {
                GpuTile g;
                for (int k = 0; k < 5; ++k) g.groups[k] = t.groups[k];
                g.vertices = t.vertices(); g.faces = t.triangles(); g.coverage = t.coverage();
                g.represented = t.represented; g.source_points = t.source_points; g.exact = t.exact();
                g.last_frame = frame_;
                gpu_[t.key] = g;
                gpu_bounds_[t.key] = {t.bmin[2], t.bmax[2]};
                gpu_vertices_ += g.vertices; gpu_faces_ += g.faces; gpu_coverage_ += g.coverage;
                ++stats_[kStatUploaded];
            } else if (descs[i].result < 0) {
                full = true;
            } else {
                std::fprintf(stderr, "[LNT surface] tile %016llx upload rejected: %s\n", (unsigned long long)t.key, eng.last_error());
                failed_.insert(t.key);
            }
        }
        if (full) gpu_pressure_ = true;
    }

    // GPU eviction: least recently drawn tiles that are not drawn now.
    void make_room(VulkanEngine& eng, const uint64_t need[3]) {
        const uint64_t used[3] = {gpu_vertices_, gpu_faces_, gpu_coverage_};
        bool over = gpu_pressure_;
        for (int i = 0; i < 3; ++i) over |= used[i] + need[i] > gpu_caps_[i] * 9 / 10;
        if (!over) return;
        std::unordered_set<uint64_t> keep(drawn_.begin(), drawn_.end());
        keep.insert(make_key(0, 0, 0)); // Continuous interpolation underlay.
        // Ready siblings and resident ancestors may have been rolled out of
        // drawn_ while another child loads. Evicting them here makes the next
        // upload evict a different sibling: refinement never completes and
        // pan/zoom repeatedly falls back to the root.
        keep.insert(cover_keep_.begin(), cover_keep_.end());
        std::vector<std::pair<uint64_t, uint64_t>> lru;
        for (const auto& g : gpu_) if (!keep.count(g.first)) lru.push_back({g.second.last_frame, g.first});
        std::sort(lru.begin(), lru.end());
        std::vector<uint64_t> victims;
        uint64_t v = gpu_vertices_, f = gpu_faces_, c = gpu_coverage_;
        const double target = gpu_pressure_ ? 0.6 : 0.75;
        for (const auto& e : lru) {
            if (v + need[0] <= gpu_caps_[0] * target && f + need[1] <= gpu_caps_[1] * target &&
                c + need[2] <= gpu_caps_[2] * target) break;
            const GpuTile& g = gpu_[e.second];
            v -= g.vertices; f -= g.faces; c -= g.coverage;
            victims.push_back(e.second);
        }
        std::vector<uint64_t> engine_keys;
        for (uint64_t k : victims) {
            if (!gpu_[k].empty) engine_keys.push_back(k);
            gpu_.erase(k);
            gpu_bounds_.erase(k);
        }
        if (!engine_keys.empty()) eng.surface_tiles_evict(engine_keys);
        gpu_vertices_ = v; gpu_faces_ = f; gpu_coverage_ = c;
        stats_[kStatGpuEvicted] += victims.size();
        gpu_pressure_ = false;
    }

    // CPU eviction: least recently used decoded tiles not waiting for an upload.
    void evict_cpu() {
        if (cpu_bytes_ <= config.cpu_bytes) return;
        std::unordered_set<uint64_t> pending(drawn_.begin(), drawn_.end());   // drawn: class edits patch them
        pending.insert(make_key(0, 0, 0)); // Keep its CPU copy for class patches.
        for (const auto& w : wanted_) if (!gpu_.count(w.second)) pending.insert(w.second);
        std::vector<std::pair<uint64_t, uint64_t>> lru;
        for (const auto& c : cpu_) if (!pending.count(c.first)) lru.push_back({c.second.last_frame, c.first});
        std::sort(lru.begin(), lru.end());
        for (const auto& e : lru) {
            if (cpu_bytes_ <= config.cpu_bytes * 8 / 10) break;
            auto it = cpu_.find(e.second);
            cpu_bytes_ -= it->second.tile->bytes();
            cpu_.erase(it);
            ++stats_[kStatCpuEvicted];
        }
    }

    // configuration / identity
    std::string path_;
    std::shared_mutex* io_lock_ = nullptr;
    uint64_t index_offset_ = 0, base_end_ = 0, total_points_ = 0;
    Quadtree qt_;
    std::unique_ptr<NodePoints> points_;
    Digest settings_{}, dir_hash_{};
    bool have_params_ = false;
    std::shared_ptr<const Snapshot> snapshot_;   // guarded by q_mu_ for workers
    std::atomic<uint64_t> params_gen_{0}, source_gen_{1};
    uint8_t lut_[768]{};
    float shade_ = 1.0f;
    bool palette_dirty_ = true;

    // archive state (a_mu_)
    std::mutex a_mu_;
    surface_archive::State arch_;
    std::unordered_map<uint64_t, surface_archive::IndexEntry> slot_entries_;
    Digest source_hash_{};
    int64_t source_mtime_ = 0;
    bool source_known_ = false, trusted_ = false;
    std::atomic<bool> verifying_{false}, verify_cancel_{false}, external_change_{false};
    std::thread verifier_;

    // work queue (q_mu_)
    mutable std::mutex q_mu_;
    std::condition_variable q_cv_;
    std::vector<Job> queue_;
    std::unordered_set<uint64_t> inflight_;
    std::vector<Completion> completions_;
    bool stop_ = true;
    std::vector<std::thread> workers_;

    // writer (w_mu_)
    std::mutex w_mu_;
    std::condition_variable w_cv_, w_done_cv_;
    std::vector<PendingItem> pending_;
    uint64_t pending_bytes_ = 0;
    std::chrono::steady_clock::time_point pending_since_{};
    bool w_stop_ = false, w_flush_ = false, w_disabled_ = false;
    std::string persist_message_, last_persist_message_;
    std::thread writer_;

    // render thread only
    uint64_t frame_ = 0;
    std::unordered_map<uint64_t, CpuTile> cpu_;
    uint64_t cpu_bytes_ = 0;
    std::unordered_map<uint64_t, GpuTile> gpu_;
    std::unordered_map<uint64_t, std::pair<double, double>> gpu_bounds_;
    uint64_t gpu_vertices_ = 0, gpu_faces_ = 0, gpu_coverage_ = 0, gpu_caps_[3]{};
    bool gpu_configured_ = false, gpu_pressure_ = false, evict_all_gpu_ = false;
    std::unordered_set<uint64_t> failed_, refined_;
    std::string failure_;
    std::vector<uint64_t> drawn_;
    std::vector<std::pair<double, uint64_t>> wanted_;
    std::unordered_set<uint64_t> cover_keep_;
    uint64_t wanted_missing_ = 0;
    std::vector<VulkanEngine::SurfaceTileDraw> last_draw_;
    std::unordered_map<uint64_t, Info> info_;
    uint64_t stats_[kStatCount]{};
    std::atomic<uint64_t> load_failures_{0}, commits_{0}, archive_bytes_{0};
};

}  // namespace surface_lod
}  // namespace lnt
