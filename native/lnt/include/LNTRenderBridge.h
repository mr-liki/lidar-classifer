// LNTRenderBridge.h - render .lnt through the existing naksha_vulkan engine.
//
// The bridge is deliberately thin, because the format and the engine already
// agree on a model: the engine's stable-offset point ARENA takes caller-chosen
// point offsets and draws an explicit list of (first, count) windows, and a .lnt
// file is a directory of nodes each owning one contiguous point chunk. So
//
//     LNTNodeEntry / node payload   ->   one arena tile
//     query_nodes_in_bounds(box)    ->   nkv_set_point_draw_ranges(box)
//
// which means panning or culling never re-uploads a point: the geometry is
// written once and the draw list changes. That is the whole point of using this
// engine rather than a bespoke renderer.
//
// PHASE 3 ADDS THE CAMERA-DRIVEN FORM OF THE SAME OPERATION, and it is the one a
// viewer uses: `LntStreamer::update_visible_nodes(engine, scene, camera, err)`
// replaces "nodes whose box intersects this query box" with "the LOD cut this
// camera can see", and replaces "whatever is already in the arena" with a lazy
// upload of exactly the cut's payloads. The box query is kept, not replaced,
// because it answers a different question with no engine and no payloads: which
// nodes are near a region (picking, export, section views). Both funnel through
// push_tiles(), so "an empty selection draws nothing" is decided in one place.
// For the same reason this file keeps plan()/reserve()/upload_tile() separate:
// the layout is a pure function of the directory, so a lazily filled arena ends
// up byte-identical to an eagerly filled one.
//
// TWO DECISIONS THE FORMAT FORCES, both of which a naive bridge gets wrong.
//
// 1. LEVELS MUST NOT BE MIXED. The directory holds full-detail LEAF nodes AND
//    strided LOD copies for interior nodes, and an interior node's points are a
//    SUBSET of its descendants' points. "Draw every node that intersects the
//    box" therefore draws most points two to four times. The bridge selects one
//    level family - leaves by default - rather than unioning them. (This is the
//    same rule the surface path already follows: one LOD level per frame, never
//    a per-tile mix.)
//
// 2. THE RENDER ORIGIN IS THE ENVELOPE CENTRE. Positions are converted to
//    float32 against the render origin, so uploading UTM-magnitude coordinates
//    with the origin left at 0 quantises every point to ~6 cm. The .lnt header
//    supplies the exact cubic envelope, so its centre is pinned BEFORE the first
//    upload. The engine would otherwise derive an origin from the first tile,
//    which depends on which tile happened to go first.
//
// A REAL LIMITATION, REPORTED RATHER THAN HIDDEN. The arena tile upload takes
// classification / intensity / normals streams but NOT an RGB stream (only the
// whole-cloud nkv_set_point_cloud carries colour). A .lnt stores r/g/b per
// point, so:
//   * the ARENA path is the streaming/culling path and shows intensity, which
//     .lnt stores faithfully;
//   * upload_bulk_rgb() shows true RGB, exercises the reader's colour fidelity
//     on screen, and gives up node culling because the whole-cloud upload
//     re-packs at offset 0.
// Both are implemented; neither pretends to be the other.

#ifndef LNT_RENDER_BRIDGE_H
#define LNT_RENDER_BRIDGE_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "LNTReader.h"
#include "LNTVulkanEngine.h"
#include "LNTTypes.h"

namespace lnt {

// One .lnt node as the engine sees it: a contiguous window in the arena, plus
// the world-space cell it occupies.
//
// The CELL is kept here rather than re-derived at query time. It cannot be
// recovered from half_size alone - a box needs a centre too - and the centre
// lives in the directory as a float32 offset from the envelope minimum, so the
// one place that value is unambiguous is where the reader has the header in
// hand. `bounds_of()` already does that conversion, so the result is stored.
struct LntTile {
    uint64_t    node_id = 0;
    BoundingBox cell{};        // world-space node cell, from the directory
    uint32_t    first = 0;     // arena point offset
    uint32_t    count = 0;     // points in this tile
    uint8_t     level = 0;     // tree_level from the directory
    bool        leaf = true;
    // Set once this tile's payload has been written to the arena at `first`.
    // The streaming path pushes draw ranges ONLY for uploaded tiles: a range over
    // a reserved-but-unwritten arena slot would rasterise uninitialised memory
    // into the frame, which looks like garbage geometry rather than a bug.
    bool        uploaded = false;
};

// Bytes the engine spends per point in the arena: float32 x3 positions,
// 1 byte classification, 4 bytes intensity. Used for the budget guard.
static constexpr uint64_t kArenaBytesPerPoint = 12 + 1 + 4 + 4;   // + RGBA8 colour

class LntScene {
public:
    // Upload every node's payload into the engine arena, one tile per node, in
    // directory order. Sets the render origin from the file's envelope first.
    // Refuses (without touching the engine) when the arena would exceed the
    // device budget or the 32-bit draw-range ABI.
    bool upload(VulkanEngine& engine, Reader& reader, std::string& err);

    // Whole-cloud upload WITH colour. Gives up node culling (the engine re-packs
    // at offset 0), which is why it is a separate entry point rather than a flag
    // on upload().
    bool upload_bulk_rgb(VulkanEngine& engine, Reader& reader, std::string& err);

    const std::vector<LntTile>& tiles() const { return tiles_; }
    const LntTile* find_tile(uint64_t node_id) const;

    uint64_t arena_points() const { return arena_points_; }
    uint64_t leaf_points() const { return leaf_points_; }
    uint64_t lod_points() const { return lod_points_; }
    uint64_t leaf_tiles() const { return leaf_tiles_; }
    uint64_t lod_tiles() const { return lod_tiles_; }
    void render_origin(double out3[3]) const {
        out3[0] = origin_[0]; out3[1] = origin_[1]; out3[2] = origin_[2];
    }
    double render_origin(int axis) const { return origin_[axis]; }

    // Directory-order tiles intersecting `box`. `leaves_only` is the safe
    // default - see decision 1 in the file header.
    std::vector<const LntTile*> tiles_in_bounds(const BoundingBox& box,
                                                bool leaves_only = true) const;

    // Push the selection to the engine as draw ranges. Returns the number of
    // ranges set (0 means "nothing selected", which in arena mode draws
    // nothing). Never uploads.
    std::size_t apply_view(VulkanEngine& engine, const BoundingBox& box,
                           bool leaves_only = true) const;
    bool show_all(VulkanEngine& engine, bool leaves_only = true) const;
    bool hide_all(VulkanEngine& engine) const;

    // ---- plan / reserve / lazy upload (the streaming path) ----------------
    // upload() below is plan + reserve + every payload. The streaming path needs
    // the three steps separately, because "lay out the whole arena" and "write
    // this node" are different events when the working set is decided by a
    // camera rather than by the file: the layout is fixed up front (a stable
    // offset per node is what the draw-range ABI requires) while the writes
    // happen as nodes come into view.
    //
    // plan() builds the tile table and the offsets from the directory alone - no
    // payload, no device. reserve() is the only device call in the layout step.
    bool plan(Reader& reader, std::string& err);
    bool planned() const { return planned_; }
    uint64_t planned_points() const { return planned_points_; }
    bool bounded_arena() const { return bounded_arena_; }
    // True when the arena wrapped: every tile is marked not uploaded again.
    bool prepare_bounded_frame(uint64_t incoming_points) {
        if (!bounded_arena_ || next_gpu_point_ + incoming_points <= gpu_capacity_points_) return false;
        for (LntTile& tile : tiles_) tile.uploaded = false;
        next_gpu_point_ = 0;
        uploaded_tiles_ = 0;
        arena_points_ = 0;
        return true;
    }
    bool reserve(VulkanEngine& engine, std::string& err);
    // Writes one tile's payload at its planned offset. Idempotent, so a payload
    // evicted and reloaded can simply be written again in place.
    bool upload_tile(VulkanEngine& engine, const std::vector<RawPoint>& pts,
                     LntTile& tile, std::string& err);
    uint64_t uploaded_tiles() const { return uploaded_tiles_; }

    // Mutable tile lookup, for the streaming path that has to record an upload.
    LntTile* tile_mut(uint64_t node_id);

    // Push an explicit (first, count) list. The one place draw ranges are set,
    // so every path - bounds query, camera cut, show-all - agrees on what an
    // empty selection means (clear, i.e. draw nothing in arena mode).
    bool push_tiles(VulkanEngine& engine, const std::vector<uint32_t>& firsts,
                    const std::vector<uint32_t>& counts) const;
    std::size_t push_selection(VulkanEngine& engine,
                               const std::vector<const LntTile*>& selection) const;

    // Point count the CURRENT selection asks the engine to draw, read back from
    // the engine - so the harness compares engine state against its own
    // expectation rather than against itself.
    uint32_t engine_draw_range_points(const VulkanEngine& engine) const {
        return engine.draw_range_points();
    }

private:
    const Reader* line_reader_ = nullptr;
    std::vector<uint8_t> line_rgb_, line_visible_;
    std::unordered_map<uint64_t, std::vector<uint8_t>> line_masks_;
public:
    void set_lines(const Reader* reader, const uint8_t* rgb, const uint8_t* visible) {
        line_reader_=reader; line_masks_.clear();
        if (reader) { line_rgb_.assign(rgb,rgb+65536*3); line_visible_.assign(visible,visible+65536); }
        for (auto& t : tiles_) t.uploaded=false;
    }
private:
    bool upload_impl(VulkanEngine& engine, Reader& reader, std::string& err,
                     bool with_rgb);

    std::vector<LntTile> tiles_;
    std::unordered_map<uint64_t, std::size_t> by_id_;
    uint64_t arena_points_ = 0;
    uint64_t leaf_points_ = 0;
    uint64_t lod_points_ = 0;
    uint64_t leaf_tiles_ = 0;
    uint64_t lod_tiles_ = 0;
    double origin_[3] = {0.0, 0.0, 0.0};
    bool planned_ = false;
    uint64_t planned_points_ = 0;
    uint64_t uploaded_tiles_ = 0;
    bool bounded_arena_ = false;
    uint64_t gpu_capacity_points_ = 0;
    uint64_t next_gpu_point_ = 0;
};

// ========================================================================
// implementations
// ========================================================================
inline const LntTile* LntScene::find_tile(uint64_t node_id) const {
    auto it = by_id_.find(node_id);
    return it == by_id_.end() ? nullptr : &tiles_[it->second];
}

inline bool LntScene::upload(VulkanEngine& engine, Reader& reader,
                             std::string& err) {
    return upload_impl(engine, reader, err, /*with_rgb=*/false);
}

inline bool LntScene::upload_bulk_rgb(VulkanEngine& engine, Reader& reader,
                                      std::string& err) {
    return upload_impl(engine, reader, err, /*with_rgb=*/true);
}

// ---- plan: the layout, from the directory alone -------------------------
// Split out of upload() in Phase 3. Offsets are the running sum of the preceding
// tiles' counts, so a node's window is a pure function of the directory and does
// not depend on upload order or on which nodes are later hidden - which is what
// lets a lazily filled arena end up byte-identical to an eagerly filled one.
inline bool LntScene::plan(Reader& reader, std::string& err) {
    tiles_.clear();
    by_id_.clear();
    arena_points_ = leaf_points_ = lod_points_ = 0;
    leaf_tiles_ = lod_tiles_ = 0;
    uploaded_tiles_ = 0;
    planned_points_ = 0;
    planned_ = false;
    bounded_arena_ = false;
    gpu_capacity_points_ = next_gpu_point_ = 0;

    if (!reader.ok()) { err = "reader is not open: " + reader.error(); return false; }
    if (reader.get_total_nodes() == 0) { err = "file has no nodes"; return false; }

    // ---- render origin, pinned from the file BEFORE any upload -------------
    const BoundingBox env = reader.get_bounds();
    if (env.is_empty()) { err = "file envelope is empty"; return false; }
    origin_[0] = env.mid(0);
    origin_[1] = env.mid(1);
    origin_[2] = env.mid(2);

    uint64_t total = 0;
    for (const LNTNodeEntry& e : reader.nodes()) total += e.point_count;
    if (total == 0) { err = "every node is empty"; return false; }
    if (total > 0xFFFFFFFFull) {
        err = "arena would need " + std::to_string(total) +
              " point offsets, past the 32-bit draw-range ABI limit";
        return false;
    }

    tiles_.reserve(reader.nodes().size());
    uint64_t off = 0;
    for (const LNTNodeEntry& e : reader.nodes()) {
        LntTile t;
        t.node_id = e.node_id;
        t.first = static_cast<uint32_t>(off);
        t.count = e.point_count;
        t.level = e.tree_level;
        t.leaf = is_leaf_node(e);
        t.cell = reader.bounds_of(e);
        by_id_[t.node_id] = tiles_.size();
        tiles_.push_back(t);
        off += e.point_count;
        if (t.count == 0) continue;
        if (t.leaf) { leaf_points_ += t.count; ++leaf_tiles_; }
        else        { lod_points_  += t.count; ++lod_tiles_; }
    }
    planned_points_ = total;
    planned_ = true;
    return true;
}

// Reserves arena capacity and pins the render origin. Separate from plan()
// because this is the step that touches the device, and a caller that only wants
// the layout - a headless selection test - must not need a live engine.
inline bool LntScene::reserve(VulkanEngine& engine, std::string& err) {
    if (!planned_) { err = "plan() must run before reserve()"; return false; }
    if (!engine.ok()) { err = "engine is not open: " + engine.error(); return false; }
    if (!engine.set_render_origin(origin_[0], origin_[1], origin_[2])) {
        err = "nkv_set_render_origin failed";
        return false;
    }
    const uint64_t budget = engine.vram_budget();
    // Large files keep only the active working set on the GPU. Leave headroom
    // for the swapchain and other Vulkan allocations on a shared device.
    const uint64_t safe_points = budget ? (budget / kArenaBytesPerPoint) * 3 / 5
                                        : planned_points_;
    gpu_capacity_points_ = std::min(planned_points_, safe_points);
    bounded_arena_ = gpu_capacity_points_ < planned_points_;
    if (gpu_capacity_points_ == 0) { err = "no GPU point capacity available"; return false; }
    const int rc = engine.reserve_point_capacity(gpu_capacity_points_);
    if (rc == 0) {
        err = std::string("nkv_reserve_point_capacity failed: ") + engine.last_error();
        return false;
    }
    if (rc == 2) {
        // A reallocation invalidates every earlier tile, which cannot happen on
        // a first reserve - so treat it as a hard error rather than silently
        // proceeding with stale offsets.
        err = "arena was reallocated on first reserve; offsets would be stale";
        return false;
    }
    return true;
}

// Writes one tile's payload at its planned offset. Idempotent: the same tile
// written twice lands in the same bytes, so a payload that was evicted and
// reloaded can be written straight back without disturbing any other tile's
// offsets - the property the stable-offset arena exists to provide.
inline bool LntScene::upload_tile(VulkanEngine& engine,
                                 const std::vector<RawPoint>& pts, LntTile& tile,
                                 std::string& err) {
    if (tile.count == 0) { tile.uploaded = true; return true; }
    if (pts.size() != tile.count) {
        err = "tile " + std::to_string(tile.node_id) + " has " +
              std::to_string(pts.size()) + " points, the directory says " +
              std::to_string(tile.count);
        return false;
    }
    std::vector<double> xyz(pts.size() * 3);
    std::vector<float> inten(pts.size());
    std::vector<uint8_t> cls(pts.size());
    std::vector<uint8_t> rgb(pts.size() * 3);
    std::vector<uint16_t> lines;
    if (line_reader_) {
        const auto* n=line_reader_->find_node(tile.node_id);
        if (!n || !line_reader_->read_line_ids(*n,lines)) { err="Flight-line sidecar is missing or invalid"; return false; }
        auto& mask=line_masks_[tile.node_id]; mask.resize(lines.size());
        for (size_t i=0;i<lines.size();++i) mask[i]=line_visible_[lines[i]];
    }
    for (std::size_t i = 0; i < pts.size(); ++i) {
        cls[i] = pts[i].classification;
        rgb[i * 3 + 0] = pts[i].r;
        rgb[i * 3 + 1] = pts[i].g;
        rgb[i * 3 + 2] = pts[i].b;
        if (line_reader_) for (int k=0;k<3;++k) rgb[i*3+k]=line_rgb_[size_t(lines[i])*3+k];
        xyz[i * 3 + 0] = pts[i].x;
        xyz[i * 3 + 1] = pts[i].y;
        xyz[i * 3 + 2] = pts[i].z;
        // .lnt stores an 8-bit intensity and the engine's intensity stream is
        // float32, so this is a widening, not a compression.
        inten[i] = float(pts[i].intensity);
    }
    // Classification is carried per point (format version 2), so the Class
    // display mode and the application's palette (.ptc) work on streamed tiles.
    if (bounded_arena_) {
        if (next_gpu_point_ + tile.count > gpu_capacity_points_) {
            err = "visible tile exceeds the bounded GPU arena";
            return false;
        }
        tile.first = static_cast<uint32_t>(next_gpu_point_);
    }
    if (!engine.upload_point_tile(tile.first, tile.count, xyz.data(), cls.data(),
                                  inten.data(), engine.has_rgb_tiles() ? rgb.data() : nullptr)) {
        err = "arena tile upload failed for node " + std::to_string(tile.node_id) +
              ": " + engine.last_error();
        return false;
    }
    tile.uploaded = true;
    if (bounded_arena_) next_gpu_point_ += tile.count;
    ++uploaded_tiles_;
    arena_points_ += tile.count;
    return true;
}

inline bool LntScene::upload_impl(VulkanEngine& engine, Reader& reader,
                                  std::string& err, bool with_rgb) {
    if (!plan(reader, err)) return false;
    if (!engine.ok()) { err = "engine is not open: " + engine.error(); return false; }

    if (with_rgb) {
        // Whole-cloud path: decode the leaves and hand the engine float64 XYZ
        // plus an interleaved uint8 RGB buffer. Node culling is not available
        // here by construction.
        // Leaves only: an interior node's chunk is a strided LOD sample of its
        // descendants, so including it would upload most points two to four
        // times.
        //
        // Decoded here rather than through the reader's read_all_leaf_points()
        // helper for a structural reason, not to duplicate it: this path needs
        // interleaved float64 XYZ and uint8 RGB buffers directly, while the
        // helper returns a RawPoint vector that would then be copied field by
        // field. The helper also lives in the split TU on purpose (see
        // LNTReader.h), and main.cpp must not depend on it or the
        // single-command harness build stops linking.
        std::vector<RawPoint> pts;
        std::vector<RawPoint> node_pts;
        uint64_t ln = 0, ldn = 0;
        for (const LNTNodeEntry& e : reader.nodes()) {
            if (!is_leaf_node(e)) { ++ldn; continue; }
            if (!reader.read_node_points(e, node_pts)) {
                err = "reading leaf node " + std::to_string(e.node_id) +
                      " failed: " + reader.last_error();
                return false;
            }
            pts.insert(pts.end(), node_pts.begin(), node_pts.end());
            ++ln;
        }
        if (pts.empty()) { err = "no leaf points to upload"; return false; }
        if (pts.size() > 0xFFFFFFFFull) {
            err = "bulk upload is limited to 32-bit point counts";
            return false;
        }
        std::vector<double> xyz(pts.size() * 3);
        std::vector<uint8_t> rgb(pts.size() * 3);
        for (std::size_t i = 0; i < pts.size(); ++i) {
            xyz[i * 3 + 0] = pts[i].x;
            xyz[i * 3 + 1] = pts[i].y;
            xyz[i * 3 + 2] = pts[i].z;
            rgb[i * 3 + 0] = pts[i].r;
            rgb[i * 3 + 1] = pts[i].g;
            rgb[i * 3 + 2] = pts[i].b;
        }
        if (!engine.set_point_cloud(xyz.data(), rgb.data(), nullptr, nullptr,
                                    pts.size())) {
            err = std::string("nkv_set_point_cloud failed: ") + engine.last_error();
            return false;
        }
        arena_points_ = pts.size();
        leaf_points_ = pts.size();
        leaf_tiles_ = ln;
        lod_tiles_ = ldn;
        // The tile table is already built by plan(); only the upload state has
        // to be recorded, because this path uploads the whole cloud at offset 0
        // and every tile is therefore "on the GPU" as far as drawing is
        // concerned - which is exactly why it gives up node culling.
        for (LntTile& t : tiles_) t.uploaded = true;
        uploaded_tiles_ = tiles_.size();
        return true;
    }

    if (!reserve(engine, err)) return false;

    // ---- one tile per node, in directory order ----------------------------
    std::vector<RawPoint> pts;
    for (LntTile& t : tiles_) {
        if (t.count == 0) { t.uploaded = true; continue; }
        if (!reader.read_node_points(t.node_id, pts)) {
            err = "node " + std::to_string(t.node_id) +
                  " payload read failed: " + reader.last_error();
            return false;
        }
        if (!upload_tile(engine, pts, t, err)) return false;
    }

    engine.set_display_mode(kEngineElevation);
    engine.set_point_size_params(0.12f, 1.5f, 8.0f, 0.0f);
    return true;
}

inline std::vector<const LntTile*>
LntScene::tiles_in_bounds(const BoundingBox& box, bool leaves_only) const {
    std::vector<const LntTile*> out;
    if (box.is_empty()) return out;
    for (const LntTile& t : tiles_) {
        if (leaves_only && !t.leaf) continue;
        if (t.cell.intersects(box)) out.push_back(&t);
    }
    return out;
}

inline bool LntScene::show_all(VulkanEngine& engine, bool leaves_only) const {
    std::vector<uint32_t> firsts, counts;
    for (const LntTile& t : tiles_) {
        if (leaves_only && !t.leaf) continue;
        if (t.count == 0) continue;
        firsts.push_back(t.first);
        counts.push_back(t.count);
    }
    if (firsts.empty()) return engine.clear_point_draw_ranges();
    return engine.set_point_draw_ranges(firsts, counts);
}

inline bool LntScene::hide_all(VulkanEngine& engine) const {
    return engine.clear_point_draw_ranges();
}

inline bool LntScene::push_tiles(VulkanEngine& engine,
                                const std::vector<uint32_t>& firsts,
                                const std::vector<uint32_t>& counts) const {
    if (firsts.size() != counts.size()) return false;
    // An empty list is not "draw everything" in arena mode - it is "draw
    // nothing". Clearing the ranges is how the engine is told that, and it is
    // the same call every path makes, so there is one meaning and not three.
    if (firsts.empty()) return engine.clear_point_draw_ranges();
    if (line_reader_) {
        std::vector<uint32_t> f,c;
        for (size_t j=0;j<firsts.size();++j) {
            const uint64_t lo=firsts[j], hi=lo+counts[j];
            for (const auto& tile : tiles_) {
                if (!tile.uploaded || tile.first>=hi || uint64_t(tile.first)+tile.count<=lo) continue;
                auto it=line_masks_.find(tile.node_id); if(it==line_masks_.end()) continue;
                const auto& mask=it->second;
                uint32_t begin=uint32_t(std::max<uint64_t>(lo,tile.first)-tile.first);
                const uint32_t end=uint32_t(std::min<uint64_t>(hi,uint64_t(tile.first)+tile.count)-tile.first);
                while(begin<end) {
                    while(begin<end && !mask[begin]) ++begin;
                    uint32_t finish=begin; while(finish<end && mask[finish]) ++finish;
                    if(finish>begin) { f.push_back(tile.first+begin); c.push_back(finish-begin); }
                    begin=finish;
                }
            }
        }
        return f.empty()?engine.clear_point_draw_ranges():engine.set_point_draw_ranges(f,c);
    }
    return engine.set_point_draw_ranges(firsts, counts);
}

inline std::size_t LntScene::push_selection(
    VulkanEngine& engine, const std::vector<const LntTile*>& selection) const {
    std::vector<uint32_t> firsts, counts;
    firsts.reserve(selection.size());
    counts.reserve(selection.size());
    for (const LntTile* t : selection) {
        if (t == nullptr || t->count == 0) continue;
        firsts.push_back(t->first);
        counts.push_back(t->count);
    }
    push_tiles(engine, firsts, counts);
    return firsts.size();
}

inline LntTile* LntScene::tile_mut(uint64_t node_id) {
    auto it = by_id_.find(node_id);
    return it == by_id_.end() ? nullptr : &tiles_[it->second];
}

inline std::size_t LntScene::apply_view(VulkanEngine& engine,
                                        const BoundingBox& box,
                                        bool leaves_only) const {
    std::vector<const LntTile*> sel;
    for (const LntTile& t : tiles_) {
        if (leaves_only && !t.leaf) continue;
        if (t.count == 0) continue;
        if (!t.cell.intersects(box)) continue;
        sel.push_back(&t);
    }
    return push_selection(engine, sel);
}

}  // namespace lnt

#endif  // LNT_RENDER_BRIDGE_H
