// LNTStreamer.h - camera-driven out-of-core streaming for .lnt.
//
// WHAT THIS ADDS. Phase 2 established two independent things: a Reader that can
// pull one node's payload off disk, and a bridge that turns a set of nodes into
// engine draw ranges without re-uploading anything. What was missing is the
// decision - WHICH nodes, for THIS camera - and the machinery to make that
// decision cost no disk time on the render thread. That is this file:
//
//     LNTCamera  +  LNTNodeTree  ->  LntLodSelection  ->  cache residency  ->  draw ranges
//
// THE HIERARCHY HAS TO BE RECONSTRUCTED, AND THAT IS NOT AN OVERSIGHT. The .lnt
// node table is FLAT: it is Morton-sorted and every entry carries its own cell,
// its children_mask and its tree_level, but there is no child offset and no
// parent pointer anywhere in the 52-byte record - the writer emits the table in
// key order and lets the key carry the structure. node_id_at() encodes the level
// in a tag bit above the Morton prefix precisely so that the parent relation
// survives as `child_id >> 3 == parent_id` (main.cpp asserts this over the real
// file). LntNodeTree builds that relation ONCE, from the directory alone, so the
// traversal below is a real octree walk instead of the linear scan of all nodes
// that query_nodes_in_bounds() has to perform.
//
// WHAT THE CUT BUYS, AND WHY IT IS NOT MERELY A SPEEDUP. Selecting the nodes to
// draw as a CUT through the tree - a node is either selected or descended into,
// never both - is what makes the draw list correct without further care. An
// interior node's payload is a strided LOD sample of its descendants' points, so
// any list containing a node and also one of its descendants draws a large
// fraction of the dataset twice. The flat path had to defend against that with a
// leaves_only flag (see LNTRenderBridge.h decision 1); a cut is disjoint by
// construction and needs no flag, while still allowing coarse nodes to stand in
// for whole subtrees - which is the entire point of streaming an LOD tree.
//
// SCREEN-SPACE ERROR. SSE = (half_size / distance) * screen_height, as
// specified: descend while the error exceeds the threshold, otherwise take the
// node. A leaf is taken regardless, so the traversal always terminates on the
// tree's own depth rather than on the threshold - a threshold of 0 therefore
// selects exactly the leaves, which is what makes the "no point is lost" check
// in the harness meaningful.
//
// AN INTERIOR NODE THAT PASSES THE TEST IS DRAWN, NOT DESCENDED INTO, and that
// is the coverage guarantee rather than an optimisation. The cut a traversal
// returns is an antichain, so every leaf of the tree is either in it or under a
// node in it: a camera pulled far enough back that a subtree root's screen-space
// error is at or below the threshold selects that root's own strided LOD chunk
// and still covers every point the subtree holds. There is therefore no distance
// at which a zoom-out can drop a section of the cloud - the only way to lose
// coverage is to drop a node that IS on screen, which is the frustum's job and
// is tested in both directions (never cull a cell strictly inside the frustum,
// never keep geometry the engine's own matrix clips).
//
// WHAT A FRAME COSTS. The cut is a pure function of (tree, camera, params), so a
// camera that did not move enough to change it does not pay for it twice: the
// reuse test below compares the new camera against the one the resident cut was
// built from, and the same idea is applied one level down to the engine handoff,
// where an unchanged list of (first, count) windows skips the
// nkv_set_point_draw_ranges call entirely. Both are counters on the stats
// struct, so "the camera was still" is a measurement rather than a claim.

#ifndef LNT_STREAMER_H
#define LNT_STREAMER_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "LNTCacheManager.h"
#include "LNTCamera.h"
#include "LNTFormat.h"
#include "LNTMath.h"
#include "LNTReader.h"
#include "LNTRenderBridge.h"
#include "LNTTypes.h"
#include "LNTVulkanEngine.h"

namespace lnt {

static constexpr uint32_t kNoParent = 0xFFFFFFFFu;

// ========================================================================
// LntNodeTree - the octree relation, recovered from the flat directory
// ========================================================================
struct LntNodeTree {
    struct Node {
        uint64_t id = 0;
        int      level = 0;
        uint32_t parent = kNoParent;      // index, not id
        std::vector<uint32_t> children;   // indices, ascending: deterministic
        uint64_t subtree_nodes = 1;       // including this one
        uint64_t subtree_points = 0;      // directory point_count sum
    };

    std::vector<Node> nodes;                                   // directory order
    std::unordered_map<uint64_t, uint32_t> index_of;
    std::vector<uint32_t> roots;
    uint64_t orphan_nodes = 0;      // a node whose parent id is absent
    uint64_t edge_count = 0;
    uint64_t leaf_count = 0;        // children_mask == 0, i.e. no children
    int max_depth = 0;

    bool empty() const { return nodes.empty(); }
    std::size_t size() const { return nodes.size(); }

    const Node* find(uint64_t id) const {
        auto it = index_of.find(id);
        return it == index_of.end() ? nullptr : &nodes[it->second];
    }

    // True when `anc` is a strict ancestor of `desc`. Uses only the id encoding:
    // shifting `desc` up by one octant per level difference must land exactly on
    // `anc`. That is the same relation the file's own ids are built from, so it
    // needs no tree walk and cannot disagree with the parent/child edges above.
    static bool is_ancestor(uint64_t anc, uint64_t desc) {
        if (anc == desc) return false;
        const int la = level_of_node_id(anc);
        const int ld = level_of_node_id(desc);
        if (ld <= la) return false;
        return (desc >> (3 * (ld - la))) == anc;
    }

    // Builds the parent/child edges and the subtree aggregates from the
    // directory. No payload is read. Returns false only for an empty table.
    //
    // Every node's cell is a child of the node whose id is `id >> 3`, and a node
    // whose parent is not in the table is a ROOT. That is not the same as
    // "level 0": a pruned or partially written file can legitimately hold a
    // subtree whose ancestors were dropped, and such a node must still be
    // traversed rather than silently ignored. So roots are discovered and
    // orphans counted, not assumed.
    bool build(const std::vector<LNTNodeEntry>& entries, std::string& err) {
        nodes.clear();
        index_of.clear();
        roots.clear();
        orphan_nodes = 0;
        edge_count = 0;
        leaf_count = 0;
        max_depth = 0;
        if (entries.empty()) { err = "node table is empty"; return false; }

        nodes.resize(entries.size());
        index_of.reserve(entries.size() * 2);
        for (std::size_t i = 0; i < entries.size(); ++i) {
            nodes[i].id = entries[i].node_id;
            nodes[i].level = int(entries[i].tree_level);
            nodes[i].subtree_points = entries[i].point_count;
            if (entries[i].children_mask == 0) ++leaf_count;
            index_of.emplace(entries[i].node_id, static_cast<uint32_t>(i));
        }
        if (index_of.size() != entries.size()) {
            err = "duplicate node_id in the directory: the parent relation is "
                  "ambiguous and a tree cannot be built";
            return false;
        }

        for (std::size_t i = 0; i < entries.size(); ++i) {
            const uint64_t id = entries[i].node_id;
            if (level_of_node_id(id) == 0) { roots.push_back(uint32_t(i)); continue; }
            const uint64_t parent_id = id >> 3;
            auto it = index_of.find(parent_id);
            if (it == index_of.end()) {
                // Structurally an orphan. It is still traversed, from the root
                // list, because dropping it would drop real geometry.
                roots.push_back(uint32_t(i));
                ++orphan_nodes;
                continue;
            }
            nodes[i].parent = it->second;
            nodes[it->second].children.push_back(uint32_t(i));
            ++edge_count;
        }

        // Subtree aggregates, from the deepest level up. Ids encode their own
        // depth, so sorting the indices by level descending is enough - no
        // recursion, and no reliance on the directory already being top-down.
        std::vector<uint32_t> order(nodes.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = uint32_t(i);
        std::sort(order.begin(), order.end(), [this](uint32_t a, uint32_t b) {
            if (nodes[a].level != nodes[b].level) return nodes[a].level > nodes[b].level;
            return a < b;
        });
        for (uint32_t i : order) {
            Node& n = nodes[i];
            for (uint32_t c : n.children) {
                n.subtree_nodes += nodes[c].subtree_nodes;
                n.subtree_points += nodes[c].subtree_points;
            }
        }

        for (const Node& n : nodes) max_depth = std::max(max_depth, n.level);
        for (Node& n : nodes)
            std::sort(n.children.begin(), n.children.end());   // directory order
        std::sort(roots.begin(), roots.end());
        return true;
    }

    std::string describe() const {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "%llu nodes (%llu leaves), %llu edges, %llu roots, %llu orphans, "
                      "depth %d",
                      (unsigned long long)nodes.size(),
                      (unsigned long long)leaf_count,
                      (unsigned long long)edge_count,
                      (unsigned long long)roots.size(),
                      (unsigned long long)orphan_nodes, max_depth);
        return std::string(buf);
    }
};

// ========================================================================
// The LOD cut
// ========================================================================
struct LntLodParams {
    // Pixels. Descend while a node's screen-space error exceeds this. 0 means
    // "always descend", i.e. the cut is exactly the leaf set.
    double sse_threshold = 90.0;
    // HYSTERESIS BAND, as a fraction of the threshold. A cut is a pure function
    // of the camera, so a camera that hovers exactly at a level boundary makes the
    // cut flip between two resolutions frame after frame: the parent is drawn,
    // then its children, then the parent - and since consecutive cuts also differ
    // in point count and in which tiles are on the GPU, that flip is visible as a
    // pulse. The band fixes it the standard way: refine while SSE exceeds the
    // threshold, but only COARSEN once SSE has fallen a full band below it, so a
    // camera oscillating inside the band keeps the cut it has.
    //
    //     split while  sse > threshold
    //     split while  sse > sse_coarsen_ratio * threshold  AND the node was split
    //                 in the previous cut (otherwise the pure rule applies)
    //
    // The second line is deliberately conditioned on the previous cut: without
    // it, a node that was DRAWN last frame would refine a band earlier than the
    // pure rule, which is the same flapping pointed the other way. 0.90 is the
    // 10% buffer; 1.0 (or any value >= 1) disables the band entirely.
    double sse_coarsen_ratio = 0.90;
    // SAFETY CAP on the cut size. A camera inside a dense region of an
    // aggressive tree could otherwise ask for an unbounded number of nodes, and
    // a frame that takes unbounded work is a frame that stalls. Hitting the cap
    // is reported (LntLodSelection::truncated), never silent. 65,536 active tiles per
    // frame: the old 4,096 truncated the cut on a deep tree, and a truncated cut leaves
    // part of the view at a coarser level than asked for.
    uint32_t max_nodes = 65536;
    // Optional hard floor: never descend past this level. -1 disables it.
    int max_level = -1;
    // PARENT RETENTION WHILE CHILDREN LOAD. When the SSE wants a split, descend
    // only if every child the frustum keeps has its payload already in the cache;
    // otherwise the parent is taken instead and the missing children are recorded
    // (and requested) so the split happens the moment they land. The cut is then
    // never finer than what the data can actually show, which is the retention the
    // request asks for, expressed in the cut rather than in the draw list.
    //
    // WHAT IT COSTS, and why it defaults OFF: this makes the cut depend on CACHE
    // STATE as well as on the camera, so a payload arriving can change the cut
    // while the camera is perfectly still - and that is not a side effect, it is
    // the mechanism (the gate opens when the data lands). Measured on this
    // repository's suite, enabling it rewrites the meaning of eight existing
    // checks that assert selection as a property of the CAMERA: "the SSE dial
    // picks the level" (a cold cache would hold the cut at the root whatever the
    // dial says), "the cut grew when the payloads arrived" (it now grows by
    // design), "a still camera re-traverses nothing" (a data-driven rebuild is
    // correct and is now counted separately), and the ones that compare two cuts
    // built at different residency. Those are real behavioural changes, not
    // bugs - which is exactly why it is a MODE rather than a silent redefinition
    // of select_lod(). Section Q exercises both modes side by side.
    //
    // Coverage does not depend on this at all: the draw-list ancestor stand-in
    // (LntDrawList) covers a cut cell whose payload has not arrived whether or not
    // the gate is on, and it remains the backstop when a payload IS resident but
    // its arena write was deferred - a case residency alone cannot see.
    bool gate_split_on_residency = false;
    // CROSS-SECTION CORRIDOR. When on, a node whose cell does not intersect
    // `corridor` is culled exactly like a node outside the frustum: it and its whole
    // subtree are skipped before any payload is requested, so a deep profile section
    // loads only the corridor (less disk I/O, less VRAM). Off by default, so every
    // existing cut is unchanged.
    bool use_corridor = false;
    BoundingBox corridor{};
    // POINT BUDGET. 0 = off (the cut is whatever the SSE threshold selects, which is
    // what every older caller gets). When set, the streamer raises the SSE threshold
    // for this frame until the cut holds at most this many points, so a zoomed-out
    // view of a 100M-point file asks for a few million points, not for all of them.
    // The raise is per frame and stateless: zooming in culls the cells off screen, the
    // cut shrinks, and the full detail threshold applies again. The budget is what ties
    // the amount of data drawn to the zoom level.
    uint64_t point_budget = 0;
};

inline bool same_box(const BoundingBox& a, const BoundingBox& b) {
    for (int i = 0; i < 3; ++i)
        if (a.min[i] != b.min[i] || a.max[i] != b.max[i]) return false;
    return true;
}

struct LntLodSelection {
    std::vector<uint64_t> node_ids;   // the cut, in DIRECTORY order
    // Nodes the SSE wanted to split but the residency gate held, because a child
    // the frustum keeps has no payload yet. Their children are what the streamer
    // has to keep requesting; the ids are what "why is this cut still coarse?"
    // answers.
    std::vector<uint64_t> gated_nodes;
    // Point counts, summed from the directory for the selected nodes. This is
    // what the frame will draw once every payload is resident: for a coarse cut
    // it is much smaller than the dataset, which is the win.
    uint64_t selected_points = 0;
    uint64_t dataset_points = 0;

    uint64_t visited = 0;            // nodes whose cell was tested against the frustum
    uint64_t culled = 0;             // nodes rejected by the frustum test
    uint64_t subtree_skipped = 0;    // nodes never even tested, inside a culled subtree
    uint64_t split = 0;              // nodes descended into instead of selected
    uint64_t gated = 0;              // splits held shut by a missing child payload
    uint64_t gated_unavailable = 0;  // ... of those, held by a child that is Failed and
                                     //     can therefore never open the gate
    uint64_t leaves_taken = 0;
    int max_depth_reached = 0;
    bool truncated = false;

    // Nodes in the file that the traversal never had to look at - the headline
    // number for "did culling actually save anything", expressed in the same
    // denominator as the file rather than as a count of visits.
    uint64_t untouched(uint64_t tree_nodes) const {
        return tree_nodes > visited + subtree_skipped
                   ? tree_nodes - visited - subtree_skipped : 0;
    }
    std::size_t cut_size() const { return node_ids.size(); }
};

// The traversal. Pure by DEFAULT: with `residency` and `previous` left null it
// reads only the directory, the tree and the camera - which is what lets the
// harness assert the cut as a property of the geometry alone.
//
// Order of the two tests matters. The FRUSTUM test comes first because it is the
// cheap one and because it is the only one that can discard a whole subtree in a
// single go: a node outside the frustum has every descendant outside it too,
// since a child's cell is contained in its parent's. Applying the SSE test first
// would compute a distance for nodes that are going to be thrown away.
//
// TWO OPTIONAL INPUTS, each defaulting to today's pure behaviour:
//   * `residency` - enables the parent-retention gate: a split is held shut until
//     every frustum-keeping child's payload is in memory (LntLodParams).
//   * `previous`  - enables the SSE coarsening band: a node that was split in the
//     previous cut keeps being split until SSE falls a full band below the
//     threshold, so a camera hovering at a level boundary does not flap.
inline LntLodSelection select_lod(const LntNodeTree& tree,
                                  const std::vector<LNTNodeEntry>& entries,
                                  const double file_origin[3],
                                  const LNTCamera& cam,
                                  const LntLodParams& params,
                                  uint64_t dataset_points = 0,
                                  const LntCacheManager* residency = nullptr,
                                  const LntLodSelection* previous = nullptr) {
    LntLodSelection sel;
    sel.dataset_points = dataset_points;

    // The coarsening band only means anything across two consecutive cuts, so the
    // marks are built only when a previous cut is supplied. They are O(cut x
    // depth), and the marks for a node exclude the node itself: `held_split[i]`
    // means "a STRICT DESCENDANT of i was drawn last frame", i.e. i was split -
    // never "i was drawn", which would widen the refine side of the band and
    // re-introduce the flapping the band exists to stop.
    const bool band = previous != nullptr && params.sse_coarsen_ratio > 0.0 &&
                      params.sse_coarsen_ratio < 1.0;
    std::vector<uint32_t> held_split;
    if (band) {
        held_split.assign(tree.nodes.size(), 0u);
        for (uint64_t id : previous->node_ids) {
            const auto it = tree.index_of.find(id);
            if (it == tree.index_of.end()) continue;
            for (uint32_t cur = tree.nodes[it->second].parent; cur != kNoParent;
                 cur = tree.nodes[cur].parent)
                held_split[cur] = 1;
        }
    }

    std::vector<uint32_t> stack;
    stack.reserve(256);
    // Reversed push so the first root is popped first, which keeps the traversal
    // order stable across runs.
    for (std::size_t i = tree.roots.size(); i-- > 0;) stack.push_back(tree.roots[i]);

    while (!stack.empty()) {
        const uint32_t idx = stack.back();
        stack.pop_back();
        if (idx >= entries.size()) continue;
        const LNTNodeEntry& e = entries[idx];
        const LntNodeTree::Node& n = tree.nodes[idx];

        ++sel.visited;
        if (!cam.visible(e, file_origin) ||
            (params.use_corridor && !node_bounds(e, file_origin).intersects(params.corridor))) {
            ++sel.culled;
            // Count the subtree the cull just avoided, but do not descend: the
            // point of the test is that the whole subtree is skipped.
            sel.subtree_skipped += n.subtree_nodes - 1;
            continue;
        }

        const BoundingBox cell = node_bounds(e, file_origin);
        const double sse = cam.sse(cell);
        const bool has_children = !n.children.empty();
        const bool level_ok = (params.max_level < 0) || (n.level < params.max_level);
        const bool room = sel.node_ids.size() < params.max_nodes;
        bool want_split = has_children && level_ok && room &&
                          (sse > params.sse_threshold);
        // The coarsening band: this node was split in the previous cut, so its
        // children are what the viewer is looking at; do not snap back to it until
        // the pure rule is not merely unmet but a full band away from being met.
        if (!want_split && has_children && level_ok && room && band &&
            held_split[idx] != 0 &&
            sse > params.sse_coarsen_ratio * params.sse_threshold)
            want_split = true;

        // PARENT RETENTION: only descend into children that can actually be drawn.
        // A child with no payload would put a hole where the parent's own points
        // were, so the whole split is held and the parent is taken instead - one
        // coarse node covering all eight children, at the cost of a few
        // milliseconds of detail rather than a black rectangle. Children the
        // frustum discards are not waited for: nothing would draw them anyway.
        if (want_split && residency != nullptr && params.gate_split_on_residency) {
            bool hold = false, unavailable = false;
            for (uint32_t c : n.children) {
                if (c >= entries.size()) continue;
                if (!cam.visible(entries[c], file_origin)) continue;
                const LntResidency r = residency->residency(entries[c].node_id);
                if (r == LntResidency::Resident) continue;
                hold = true;
                if (r == LntResidency::Failed) unavailable = true;   // can never arrive
            }
            if (hold) {
                want_split = false;
                sel.gated_nodes.push_back(e.node_id);
                ++sel.gated;
                if (unavailable) ++sel.gated_unavailable;
            }
        }

        if (want_split) {
            ++sel.split;
            for (std::size_t i = n.children.size(); i-- > 0;)
                stack.push_back(n.children[i]);
            continue;
        }
        if (has_children && level_ok && sse > params.sse_threshold && !room)
            sel.truncated = true;

        sel.node_ids.push_back(e.node_id);
        sel.selected_points += e.point_count;
        if (!has_children) ++sel.leaves_taken;
        sel.max_depth_reached = std::max(sel.max_depth_reached, n.level);
    }

    // Directory order. The stack walk is depth-first and therefore already
    // deterministic, but sorting by the directory's own order makes the draw list
    // spatially coherent (the table is Morton-sorted) and independent of the
    // traversal order - which matters because draw-range order is observable.
    if (!tree.index_of.empty()) {
        std::sort(sel.node_ids.begin(), sel.node_ids.end(),
                  [&tree](uint64_t a, uint64_t b) {
                      auto ia = tree.index_of.find(a);
                      auto ib = tree.index_of.find(b);
                      const uint32_t va = ia == tree.index_of.end() ? 0xFFFFFFFFu : ia->second;
                      const uint32_t vb = ib == tree.index_of.end() ? 0xFFFFFFFFu : ib->second;
                      return va < vb;
                  });
    }
    return sel;
}

// ========================================================================
// The draw list: what the engine is told to draw, which is not the cut
// ========================================================================
// THE CUT IS A PLAN; THE DRAW LIST IS WHAT CAN BE SHOWN THIS FRAME. A cut node
// is drawable only once its payload is in memory AND has been written to the
// arena, and both steps are asynchronous: the loader thread decides when a
// payload arrives, and the per-frame upload time slice decides when it reaches
// the GPU. Between the two, the node's cell has nothing on the GPU at all.
//
// DROPPING SUCH A NODE IS WHAT MAKES A ZOOM SHOW RECTANGULAR VOIDS, and the zoom
// band where it happens is not a coincidence. When the cut descends, a parent is
// replaced by its children IN THE SAME FRAME, and the children are exactly the
// nodes whose payloads were just requested - so for every frame between the
// split and the arrival of the payloads, the parent's entire cell is blank. Not
// a soft edge and not a sparse region: a rectangular hole the size of an octant,
// at the one distance where an octant is still large on screen. The same thing
// happens one step later, when the arena write is deferred by the upload budget.
//
// So the draw list keeps a STAND-IN. An interior node's payload is a strided LOD
// sample of its descendants' points (LNTRenderBridge.h decision 1), which means
// an ancestor's tile covers the SAME REGION at coarser density - not another
// region, not another projection of it. When a cut node has nothing on the GPU,
// the nearest ancestor that DOES is drawn in its place: the cell shows LOD
// density immediately and refines to full density the moment the child's tile
// lands. Coverage is what must not be lost; density is allowed to lag.
//
// WHAT THE FALLBACK DOES NOT DO. It never draws a tile whose payload was never
// uploaded, it never re-uploads a tile that is already in the arena, and it never
// invents coverage: a cut node with no drawable ancestor is counted in
// `uncovered_nodes`/`uncovered_points` rather than hidden. On a cold start -
// nothing on the GPU yet - every cut node is uncovered for one frame, which is
// the honest state of a renderer that has not uploaded anything, not a defect in
// the fallback.
//
// MEASURED, on this repository's own fixture (test_zoom_single.lnt.fine, 617
// nodes, depth 4), with a model of the arena in which a requested payload lands
// on the next frame: over a 2000 m -> 150 m zoom path, 392 cut cells in 4 of 24
// frames had nothing on the GPU, carrying 254 563 points of the file, and 360 of
// those 392 cells had an ancestor that was already uploaded. In other words all
// but the unavoidable first frame's cells were coverable by a tile the arena
// already held. The 32 that were not are that first frame, where the arena is
// empty by definition.
struct LntDrawList {
    // Arena windows, in cut order: one entry per window the engine must draw.
    std::vector<uint32_t> firsts;
    std::vector<uint32_t> counts;
    // Parallel to `firsts`: the node each window belongs to, and the cut node it
    // represents. They differ exactly when the window is a stand-in.
    std::vector<uint64_t> ids;
    std::vector<uint64_t> covers;
    uint64_t points = 0;              // total points these windows draw
    uint64_t stand_ins = 0;           // windows borrowed from an ancestor
    uint64_t stand_in_points = 0;     // ... their point counts
    uint64_t uncovered_nodes = 0;     // cut nodes with nothing drawable at all
    uint64_t uncovered_points = 0;    // ... the points they stand for

    std::size_t size() const { return firsts.size(); }
    bool empty() const { return firsts.empty(); }
    void clear() {
        firsts.clear(); counts.clear(); ids.clear(); covers.clear();
        points = stand_ins = stand_in_points = uncovered_nodes = uncovered_points = 0;
    }
};

// ========================================================================
// LntStreamer
// ========================================================================
// Per-frame counters. Kept on the object rather than recomputed by the caller so
// the harness reports what the streamer actually did, not a parallel guess.
struct LntStreamStats {
    uint64_t frames = 0;
    uint64_t cut_nodes = 0;          // last frame
    uint64_t cut_points = 0;         // last frame: resident+uploaded points pushed
    uint64_t wanted_points = 0;      // last frame: the cut's full point count
    uint64_t resident_cut_nodes = 0; // cut nodes already resident when the frame ran
    uint64_t requested = 0;          // cut nodes handed to the cache (cumulative enqueues)
    uint64_t ranges_pushed = 0;      // last frame
    // Traversals actually run, versus frames that reused the previous cut because
    // the camera moved (or rotated) less than the hysteresis budget. This is the
    // direct evidence that a stationary or gently moving camera costs nothing:
    // selection_runs stays put while frames climbs.
    uint64_t selection_runs = 0;
    uint64_t selection_reuses = 0;
    // WHY the cut was rebuilt: `camera` for the motion/rotation/parameter rules,
    // `data` when a payload the residency gate was holding for arrived. Splitting
    // them is what keeps "a still camera re-traverses nothing" a true statement
    // under parent retention: a data-driven rebuild is correct (the cut must open
    // the moment the data lands, or a stopped camera stays blocky for ever), while
    // camera-driven ones are the thing the motion hysteresis exists to prevent.
    uint64_t selection_runs_camera = 0;
    uint64_t selection_runs_data = 0;
    // Parent retention, last frame and cumulative: how many splits the residency
    // gate held shut because a visible child had no payload yet, and how many of
    // those were held by a child that Failed (and can therefore never open the
    // gate - a coarse but VISIBLE parent, reported rather than silent).
    uint64_t gated_cells = 0;
    uint64_t gated_frames = 0;           // frames on which the gate held anything
    uint64_t gated_unavailable_cells = 0;
    uint64_t gated_children_requested = 0;
    // Draw-list handoffs. nkv_set_point_draw_ranges carries the whole list in one
    // call, so these count CALLS, not ranges: `range_pushes` are the frames that
    // actually re-uploaded the windows and `range_pushes_skipped` the frames that
    // found them identical and left the engine's list alone (cumulative).
    uint64_t range_pushes = 0;
    uint64_t range_pushes_skipped = 0;
    uint64_t tiles_uploaded = 0;     // cumulative lazy uploads
    uint64_t uploads_deferred = 0;   // resident tiles held back by the upload budget (cumulative)
    // ---- coverage of the frame's draw list (see LntDrawList) --------------
    // A non-zero stand_in count is the fallback working: this many cut cells were
    // drawn from an ancestor's tile instead of going blank while their own
    // payload streams in or waits for the upload slice.
    uint64_t stand_in_ranges = 0;    // last frame
    uint64_t stand_in_points = 0;    // last frame
    uint64_t stand_in_frames = 0;    // frames that used the fallback at least once
    bool     holding = false;        // this frame kept the previous complete draw list
    uint64_t held_frames = 0;        // frames that did so (cumulative)
    uint64_t uncovered_cells = 0;    // last frame: cut cells with nothing on the GPU
    uint64_t uncovered_cells_total = 0;   // cumulative, including the cold start
    uint64_t max_uncovered_cells = 0;     // worst frame, so a single bad frame cannot hide
    // Time the MAIN thread spent inside update_visible_nodes(), in microseconds.
    // Measured here because this is the only place the render thread could be
    // blocked by streaming at all.
    uint64_t main_ns = 0;
    uint64_t max_main_ns = 0;
    uint64_t truncated_frames = 0;
    LntLodSelection last_selection;
};

class LntStreamer {
public:
    LntStreamer() = default;

    // Builds the hierarchy from a directory. `cache` may be null for a pure
    // selection-only streamer (the harness uses that to test the cut in
    // isolation). No payload is read here.
    inline bool attach(const Reader& reader, LntCacheManager* cache, std::string& err) {
        if (!reader.ok()) { err = "reader is not open: " + reader.error(); return false; }
        if (!tree_.build(reader.nodes(), err)) return false;
        entries_ = &reader.nodes();
        for (int i = 0; i < 3; ++i) origin_[i] = reader.header().bounds_min[i];
        dataset_points_ = reader.get_total_points();
        cache_ = cache;
        selection_valid_ = false;
        base_ids_.clear();
        has_previous_ = false;   // no cut has been shown yet, so no band either
        draw_ranges_valid_ = false;
        last_firsts_.clear();
        last_counts_.clear();
        last_cut_complete_ = false;
        stalled_holds_ = 0;
        stats_.holding = false;
        err.clear();
        return true;
    }

    bool attached() const { return entries_ != nullptr; }
    const LntNodeTree& tree() const { return tree_; }

    void set_params(const LntLodParams& p) {
        if (p.sse_threshold != params_.sse_threshold || p.max_nodes != params_.max_nodes ||
            p.max_level != params_.max_level ||
            p.sse_coarsen_ratio != params_.sse_coarsen_ratio ||
            p.gate_split_on_residency != params_.gate_split_on_residency ||
            p.use_corridor != params_.use_corridor || p.point_budget != params_.point_budget ||
            (p.use_corridor && !same_box(p.corridor, params_.corridor))) selection_valid_ = false;
        params_ = p;
    }
    const LntLodParams& params() const { return params_; }

    // COARSE BASE LAYER. Tiles of tree level <= `levels` are always requested first,
    // pinned in the cache and uploaded to the GPU, so every cell of the cloud always
    // has a low-density stand-in (see LntDrawList). Without it a cell whose own tile has
    // not arrived has no ancestor on the GPU either and is drawn BLACK until it does.
    // -1 turns the layer off (the default, so older callers are unchanged).
    void set_base_levels(int levels) {
        base_levels_ = levels;
        base_ids_.clear();
        if (!attached() || levels < 0) return;
        for (std::size_t i = 0; i < tree_.nodes.size(); ++i)
            if (int(tree_.nodes[i].level) <= levels) base_ids_.push_back(tree_.nodes[i].id);
        std::sort(base_ids_.begin(), base_ids_.end(), [this](uint64_t a, uint64_t b) {
            return tree_.nodes[tree_.index_of.at(a)].level < tree_.nodes[tree_.index_of.at(b)].level;
        });
    }
    // PREFETCH ALONG THE CAMERA MOTION. The cut is also computed for where the camera
    // is HEADING (its per-selection displacement, extrapolated `frames` selections ahead)
    // and the tiles that view would need are requested at LOW priority, after everything
    // the current view needs. A pan or a zoom then finds its tiles already resident.
    // 0 (the default) turns it off.
    void set_prefetch_ahead(double frames) { prefetch_ahead_ = frames; }
    // Coalesce contiguous (first, count) draw windows into one range before pushing, so
    // a cut made of many neighbouring tiles costs one draw call per run, not one per
    // tile. Off by default (it changes the engine's range count).
    void set_merge_draw_ranges(bool on) { merge_ranges_ = on; draw_ranges_valid_ = false; }
    // BACKGROUND REFINEMENT. While the new cut still has cells whose tiles are not
    // on the GPU (drawn by a coarser ancestor stand-in, or blank), keep drawing the
    // previous COMPLETE draw list and swap to the new one in a single frame once
    // every cell is uploaded. Uploads continue underneath, so the patchwork of
    // mixed densities is never shown. A cold start (nothing complete yet) and a
    // bounded-arena wrap (which overwrites the old slots) still draw progressively.
    void set_hold_incomplete_cut(bool on) { hold_incomplete_ = on; }
    bool hold_incomplete_cut() const { return hold_incomplete_; }
    uint64_t prefetch_requested() const { return prefetch_requested_; }
    static void merge_contiguous(const std::vector<uint32_t>& firsts,
                                 const std::vector<uint32_t>& counts,
                                 std::vector<uint32_t>& out_f, std::vector<uint32_t>& out_c) {
        std::vector<std::pair<uint32_t, uint32_t>> v;
        v.reserve(firsts.size());
        for (std::size_t i = 0; i < firsts.size(); ++i) v.emplace_back(firsts[i], counts[i]);
        std::sort(v.begin(), v.end());
        out_f.clear();
        out_c.clear();
        for (const auto& r : v) {
            if (!out_f.empty() && uint64_t(out_f.back()) + out_c.back() == r.first)
                out_c.back() += r.second;
            else { out_f.push_back(r.first); out_c.push_back(r.second); }
        }
    }
    // The SSE threshold the last cut actually used (>= the configured one when the point
    // budget had to raise it).
    double effective_sse() const { return effective_sse_; }
    uint64_t requests_cancelled() const { return requests_cancelled_; }

    // Restrict the cut to a cross-section corridor (world AABB), or lift it.
    void set_corridor(const BoundingBox& corridor_aabb) {
        LntLodParams p = params_;
        p.use_corridor = true;
        p.corridor = corridor_aabb;
        set_params(p);
    }
    void clear_corridor() {
        LntLodParams p = params_;
        p.use_corridor = false;
        set_params(p);
    }

    // Every directory node whose cell intersects the corridor: a directory scan, no
    // payload read. This is the set the corridor-restricted cut can draw from; a node
    // outside it is never selected, never requested and never uploaded.
    std::vector<LNTNodeEntry> query_nodes_in_corridor(const BoundingBox& corridor_aabb) const {
        std::vector<LNTNodeEntry> out;
        if (!attached() || corridor_aabb.is_empty()) return out;
        for (const LNTNodeEntry& e : *entries_)
            if (node_bounds(e, origin_).intersects(corridor_aabb)) out.push_back(e);
        return out;
    }

    // Per-frame time slice for GPU tile uploads, in milliseconds. 0 (default) =
    // unlimited, which is what every earlier caller gets. With a budget, once a
    // frame has spent it, the remaining resident-but-not-yet-uploaded tiles of the
    // cut are DEFERRED to later frames - they stay resident and nothing is lost;
    // they are simply not drawn until uploaded - which is what bounds the frame
    // when a camera move makes dozens of tiles newly visible at once. At least one
    // tile is always uploaded per frame, so the cut always makes progress.
    void set_upload_budget_ms(double ms) { upload_budget_ms_ = ms; }
    double upload_budget_ms() const { return upload_budget_ms_; }
    const LntStreamStats& stats() const { return stats_; }

    // ---- the ancestor stand-in -------------------------------------------
    // On by default. It can be switched off so the harness can prove that the
    // coverage checks built on it have teeth: with it off, the same frame that
    // reports zero uncovered cells reports the blank ones instead.
    void set_ancestor_fallback(bool on) { ancestor_fallback_ = on; }
    bool ancestor_fallback() const { return ancestor_fallback_; }

    // The windows the engine must draw for the CURRENT cut, given what `scene`
    // already has on the GPU. No disk, no device, no camera: it reads the
    // resident cut and the tiles' uploaded flags, so the harness can drive it
    // directly against a planned scene. Fills a member rather than returning one,
    // so the per-frame cost is a clear() and a few pushes after the first call.
    inline const LntDrawList& resolve_draw_list(const LntScene& scene) {
        draw_list_.clear();
        ++stamp_gen_;
        if (stamp_gen_ == 0) {                       // wrapped: start the stamps over
            std::fill(stamp_.begin(), stamp_.end(), 0u);
            stamp_gen_ = 1;
        }
        if (!attached() || !scene.planned()) return draw_list_;
        for (uint64_t id : stats_.last_selection.node_ids) {
            const auto it = tree_.index_of.find(id);
            if (it == tree_.index_of.end()) continue;
            // 1. this node's own tile, when it is on the GPU.
            const LntTile* own = scene.find_tile(id);
            if (own != nullptr && own->count > 0 && own->uploaded) {
                add_window(*own, id, id, false);
                continue;
            }
            // 2. otherwise the nearest ancestor that IS on the GPU. Its payload
            //    is a strided sample of this node's region, so drawing it there
            //    is a coarser view of the cell rather than a hole. One stand-in
            //    window covers every cut node under it, so it is pushed once and
            //    the later nodes it rescues cost a stamp test.
            const LntTile* stand = ancestor_fallback_
                                       ? first_uploaded_ancestor(scene, it->second)
                                       : nullptr;
            if (stand != nullptr) {
                const auto sit = tree_.index_of.find(stand->node_id);
                const bool pushed = (sit != tree_.index_of.end()) && stamp_seen(sit->second);
                if (!pushed) add_window(*stand, stand->node_id, id, true);
                continue;
            }
            // 3. nothing on the GPU covers this cell. Counted, never hidden.
            ++draw_list_.uncovered_nodes;
            draw_list_.uncovered_points += (own != nullptr) ? own->count : 0;
        }
        return draw_list_;
    }
    const LntDrawList& draw_list() const { return draw_list_; }

    // ---- the frame step, without the engine -------------------------------
    // Camera -> cut -> cache. Returns the number of nodes in the cut. Never
    // touches the disk: payloads that are missing are requested from the worker
    // and simply are not yet resident when this returns.
    inline std::size_t update_visible_nodes(const LNTCamera& cam) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!attached()) return 0;

        bool refused_for_data = false;
        if (can_reuse_selection(cam, &refused_for_data)) {
            ++stats_.selection_reuses;
        } else {
            ++stats_.selection_runs;
            if (refused_for_data) ++stats_.selection_runs_data;
            else                  ++stats_.selection_runs_camera;
            // The cache and the previous cut are both optional inputs: a
            // selection-only streamer (attach with a null cache) gets the pure
            // function, and so does the first frame of any streamer.
            LntLodParams run = params_;
            stats_.last_selection =
                select_lod(tree_, *entries_, origin_, cam, run, dataset_points_,
                           params_.gate_split_on_residency ? cache_ : nullptr,
                           // The PREVIOUS cut survives an invalidation: a param
                           // change (the SSE dial, for instance) forces a rebuild
                           // but does not stop the viewer from having been shown
                           // the cut below - and the coarsening band is a statement
                           // about what was on screen, so it must see it. Only
                           // attach() clears it.
                           has_previous_ ? &selection_prev_ : nullptr);
            // Point budget: coarsen until the cut fits. The traversal only visits the
            // nodes the frustum keeps, so a few extra passes are cheap next to one tile
            // read - and each pass is a pure function of the camera, so there is no
            // state to unwind when the camera moves again.
            if (params_.point_budget > 0) {
                int guard = 0;
                while (stats_.last_selection.selected_points > params_.point_budget &&
                       guard++ < 48) {
                    run.sse_threshold = std::max(run.sse_threshold * 1.3, 1.0);
                    stats_.last_selection =
                        select_lod(tree_, *entries_, origin_, cam, run, dataset_points_,
                                   params_.gate_split_on_residency ? cache_ : nullptr,
                                   has_previous_ ? &selection_prev_ : nullptr);
                }
            }
            effective_sse_ = run.sse_threshold;
            prefetch_ids_.clear();
            if (prefetch_ahead_ > 0.0 && cache_ != nullptr && has_previous_ &&
                selection_camera_.frustum_valid && !cam.from_engine_mvp) {
                LNTCamera ahead = cam;
                bool moved = false;
                for (int i = 0; i < 3; ++i) {
                    const double dt = cam.target[i] - selection_camera_.target[i];
                    const double de = cam.eye[i] - selection_camera_.eye[i];
                    if (std::fabs(dt) > 1e-9 || std::fabs(de) > 1e-9) moved = true;
                    ahead.target[i] += dt * prefetch_ahead_;
                    ahead.eye[i] += de * prefetch_ahead_;
                }
                if (moved && ahead.rebuild()) {
                    LntLodParams pp = run;
                    const LntLodSelection fut = select_lod(tree_, *entries_, origin_, ahead, pp,
                                                           dataset_points_, nullptr, nullptr);
                    std::unordered_set<uint64_t> now(stats_.last_selection.node_ids.begin(),
                                                     stats_.last_selection.node_ids.end());
                    for (uint64_t id : fut.node_ids) {
                        if (prefetch_ids_.size() >= 512) break;
                        if (!now.count(id)) prefetch_ids_.push_back(id);
                    }
                }
            }
            selection_prev_ = stats_.last_selection;
            has_previous_ = true;
            selection_camera_ = cam;
            selection_valid_ = true;
            selection_loads_ = (cache_ != nullptr) ? cache_->loads() : 0;
            min_selected_node_size_ = 0.0;
            for (uint64_t id : stats_.last_selection.node_ids) {
                const auto it = tree_.index_of.find(id);
                if (it == tree_.index_of.end()) continue;
                const double size = 2.0 * double((*entries_)[it->second].half_size);
                if (size > 0 && (min_selected_node_size_ == 0 ||
                                 size < min_selected_node_size_))
                    min_selected_node_size_ = size;
            }
        }
        const LntLodSelection& sel = stats_.last_selection;
        stats_.cut_nodes = sel.node_ids.size();
        stats_.wanted_points = sel.selected_points;
        if (sel.truncated) ++stats_.truncated_frames;
        stats_.gated_cells = sel.gated;
        stats_.gated_unavailable_cells = sel.gated_unavailable;
        if (sel.gated > 0) ++stats_.gated_frames;

        // Pin the cut BEFORE requesting: the payloads this frame is about to
        // draw are exactly the ones the eviction pass must not take, and
        // requesting first would open a window in which a payload this frame
        // needs is evicted to make room for another one it also needs.
        if (cache_ != nullptr) {
            if (base_levels_ >= 0 && base_ids_.empty()) set_base_levels(base_levels_);
            {
                std::vector<uint64_t> pins = sel.node_ids;
                pins.insert(pins.end(), base_ids_.begin(), base_ids_.end());
                cache_->set_pinned(pins);
            }
            uint64_t resident = 0;
            std::vector<uint64_t> missing;
            missing.reserve(sel.node_ids.size());
            for (uint64_t id : sel.node_ids) {
                if (cache_->residency(id) == LntResidency::Resident) ++resident;
                else missing.push_back(id);
            }
            // A gate is a promise to refine as soon as the data lands, so the held
            // node's children have to be in flight - otherwise nothing would ever
            // open the gate and a stopped camera would stay coarse for ever.
            // request() itself is a no-op for anything already queued, loading,
            // resident or Failed, so this cannot spin; the ids are the cache's own
            // directory ids, the same ones the cut uses.
            if (!sel.gated_nodes.empty()) {
                const std::size_t before = missing.size();
                for (uint64_t gid : sel.gated_nodes) {
                    const auto it = tree_.index_of.find(gid);
                    if (it == tree_.index_of.end()) continue;
                    for (uint32_t c : tree_.nodes[it->second].children) {
                        if (c >= entries_->size()) continue;
                        const LntResidency r = cache_->residency((*entries_)[c].node_id);
                        if (r == LntResidency::Resident || r == LntResidency::Failed) continue;
                        missing.push_back((*entries_)[c].node_id);
                    }
                }
                stats_.gated_children_requested += missing.size() - before;
            }
            stats_.resident_cut_nodes = resident;
            // Base layer first, then each missing tile's PARENT, then the tiles
            // themselves: a coarse stand-in lands before the detail it stands in for.
            std::vector<uint64_t> order;
            order.reserve(base_ids_.size() + missing.size() * 2);
            for (uint64_t id : base_ids_)
                if (cache_->residency(id) != LntResidency::Resident) order.push_back(id);
            std::vector<uint64_t> parents;
            for (uint64_t id : missing) {
                const auto it = tree_.index_of.find(id);
                if (it == tree_.index_of.end()) continue;
                const uint32_t pi = tree_.nodes[it->second].parent;
                if (pi == kNoParent || pi >= entries_->size()) continue;
                const uint64_t pid = (*entries_)[pi].node_id;
                if (cache_->residency(pid) != LntResidency::Resident) parents.push_back(pid);
            }
            std::sort(parents.begin(), parents.end());
            parents.erase(std::unique(parents.begin(), parents.end()), parents.end());
            auto level_of = [this](uint64_t id) {
                const auto it = tree_.index_of.find(id);
                return it == tree_.index_of.end() ? 255 : int(tree_.nodes[it->second].level);
            };
            std::stable_sort(parents.begin(), parents.end(),
                             [&](uint64_t a, uint64_t b) { return level_of(a) < level_of(b); });
            std::stable_sort(missing.begin(), missing.end(),
                             [&](uint64_t a, uint64_t b) { return level_of(a) < level_of(b); });
            order.insert(order.end(), parents.begin(), parents.end());
            order.insert(order.end(), missing.begin(), missing.end());
            // Everything the NEW view wants; requests left over from the old view that
            // are not in it are cancelled instead of being read for nothing.
            for (uint64_t id : prefetch_ids_)
                if (cache_->residency(id) != LntResidency::Resident) {
                    order.push_back(id);
                    ++prefetch_requested_;
                }
            std::unordered_set<uint64_t> wanted(order.begin(), order.end());
            for (uint64_t id : sel.node_ids) wanted.insert(id);
            // Only requests THIS streamer issued are cancelled; anyone else's stay.
            requests_cancelled_ += cache_->cancel_unwanted(wanted, issued_);
            for (auto it = issued_.begin(); it != issued_.end();)
                it = wanted.count(*it) ? std::next(it) : issued_.erase(it);
            stats_.requested += missing.size();
            cache_->request_all(order);
            for (uint64_t id : order) issued_.insert(id);
            cache_->prioritize(order);
        }

        const auto t1 = std::chrono::steady_clock::now();
        const uint64_t ns = uint64_t(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        stats_.main_ns = ns;
        if (ns > stats_.max_main_ns) stats_.max_main_ns = ns;
        ++stats_.frames;
        return sel.node_ids.size();
    }

    // ---- the frame step, with the engine ----------------------------------
    // Adds the two GPU steps to the above:
    //   3. UPLOAD the cut's payloads that have just become resident and have
    //      never been on the GPU. Phase 2's upload() writes every node eagerly;
    //      this writes only what has been looked at, which is the difference
    //      between a 40-node file and a 40-million-node one.
    //   4. PUSH the (first, count) windows of the tiles that are BOTH uploaded
    //      and on screen. Ranges are pushed only for uploaded tiles: a range over
    //      an arena slot that was reserved but never written would rasterise
    //      uninitialised memory into the frame.
    // Returns the number of draw ranges pushed.
    inline std::size_t update_visible_nodes(VulkanEngine& engine, LntScene& scene,
                                           const LNTCamera& cam, std::string& err) {
        // The WHOLE frame call is what a caller's frame budget pays for, so this
        // is what main_ns reports for this overload: selection, the cache handoff,
        // the arena uploads of newly resident tiles and the draw-range push. The
        // inner selection-only overload times itself as well, and this guard
        // deliberately overwrites that narrower number on the way out - a frame
        // that spends 6 ms uploading payloads is not a frame that cost 60 us.
        const FrameTimerCall timer(&stats_);
        const std::size_t cut = update_visible_nodes(cam);
        if (!attached()) return 0;
        if (!scene.planned()) { err = "scene is not planned: call LntScene::plan() first"; return 0; }

        // 3. lazy upload of the cut's OWN payloads. A resident payload is written
        //    to its stable arena offset; one that is still streaming in is simply
        //    not written yet. Uploading is deliberately limited to the cut: an
        //    ancestor whose payload happens to be resident is NOT uploaded here,
        //    because peeking it would mean pinning it too, and the fallback below
        //    already covers the case that matters - an ancestor whose tile is
        //    already on the GPU, which costs nothing at all to draw.
        const auto upload_t0 = std::chrono::steady_clock::now();
        if (scene.bounded_arena()) {
            uint64_t incoming_points = 0;
            for (uint64_t id : stats_.last_selection.node_ids) {
                const LntTile* tile = scene.find_tile(id);
                if (tile != nullptr && !tile->uploaded && cache_ != nullptr &&
                    cache_->residency(id) == LntResidency::Resident)
                    incoming_points += tile->count;
            }
            for (uint64_t id : base_ids_) {      // the coarse base layer is uploaded too
                const LntTile* tile = scene.find_tile(id);
                if (tile != nullptr && !tile->uploaded && cache_ != nullptr &&
                    cache_->residency(id) == LntResidency::Resident)
                    incoming_points += tile->count;
            }
            // A wrap rewrites every slot from 0: the held list would point at
            // other tiles' points, so there is nothing safe to keep drawing.
            if (scene.prepare_bounded_frame(incoming_points)) draw_ranges_valid_ = false;
        }
        uint64_t uploaded_now = 0;
        // The base layer goes first (it is pinned, so peek() is safe): coarse cover
        // before detail, so a view never has a cell with nothing to draw in it.
        for (uint64_t id : base_ids_) {
            LntTile* tile = scene.tile_mut(id);
            if (tile == nullptr || tile->count == 0 || tile->uploaded) continue;
            const std::vector<RawPoint>* pts = (cache_ != nullptr) ? cache_->peek(id) : nullptr;
            if (pts == nullptr) continue;
            if (upload_budget_ms_ > 0.0 && uploaded_now > 0 &&
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - upload_t0).count() >= upload_budget_ms_) {
                ++stats_.uploads_deferred;
                break;
            }
            if (!scene.upload_tile(engine, *pts, *tile, err)) return 0;
            ++stats_.tiles_uploaded;
            ++uploaded_now;
        }
        for (uint64_t id : stats_.last_selection.node_ids) {
            LntTile* tile = scene.tile_mut(id);
            if (tile == nullptr || tile->count == 0) continue;
            if (tile->uploaded) continue;
            const std::vector<RawPoint>* pts =
                (cache_ != nullptr) ? cache_->peek(id) : nullptr;
            if (pts == nullptr) continue;              // still streaming in
            if (upload_budget_ms_ > 0.0 && uploaded_now > 0 &&
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - upload_t0).count() >= upload_budget_ms_) {
                ++stats_.uploads_deferred;
                continue;
            }
            if (!scene.upload_tile(engine, *pts, *tile, err)) return 0;
            ++stats_.tiles_uploaded;
            ++uploaded_now;
        }

        // 4. resolve what the engine must draw, then push it. The list is NOT the
        //    cut: a cut node that has nothing on the GPU is represented by the
        //    nearest ancestor that does, so a cell whose payload is on its way is
        //    drawn at LOD density instead of going blank. See LntDrawList.
        const LntDrawList& list = resolve_draw_list(scene);
        stats_.holding = false;
        // A cut that stops making progress (nothing uploaded, nothing loading: a tile
        // that cannot be read) is shown as it is after kMaxStalledHolds frames rather
        // than held forever.
        constexpr int kMaxStalledHolds = 120;
        const bool incomplete = list.empty() || list.stand_ins > 0 || list.uncovered_nodes > 0;
        const bool progressing = uploaded_now > 0 || (cache_ != nullptr && cache_->pending() > 0);
        stalled_holds_ = (!incomplete || progressing) ? 0 : stalled_holds_ + 1;
        if (hold_incomplete_ && incomplete && draw_ranges_valid_ && last_cut_complete_ && !last_firsts_.empty() &&
            stalled_holds_ <= kMaxStalledHolds) {
            stats_.holding = true;
            ++stats_.held_frames;
            stats_.cut_nodes = cut;
            return last_firsts_.size();
        }
        if (list.empty()) {
            // Nothing drawable. Either the cut is empty or every cell in it is
            // blank - a cold start, where nothing has reached the GPU yet. The
            // engine is told to draw nothing rather than something stale, and
            // uncovered_cells is what says which case this was.
            if (!draw_ranges_valid_ || !last_firsts_.empty())
                engine.clear_point_draw_ranges();
            last_firsts_.clear();
            last_counts_.clear();
            draw_ranges_valid_ = true;
            last_cut_complete_ = false;
            stats_.ranges_pushed = 0;
            stats_.cut_points = 0;
            stats_.stand_in_ranges = 0;
            stats_.stand_in_points = 0;
            stats_.uncovered_cells = list.uncovered_nodes;
            stats_.uncovered_cells_total += list.uncovered_nodes;
            stats_.max_uncovered_cells = std::max(stats_.max_uncovered_cells,
                                                  list.uncovered_nodes);
            stats_.cut_nodes = cut;
            return 0;
        }
        if (!draw_ranges_valid_ || list.firsts != last_firsts_ ||
            list.counts != last_counts_) {
            bool pushed_ok;
            if (merge_ranges_) {
                merge_contiguous(list.firsts, list.counts, merged_f_, merged_c_);
                pushed_ok = scene.push_tiles(engine, merged_f_, merged_c_);
            } else {
                pushed_ok = scene.push_tiles(engine, list.firsts, list.counts);
            }
            if (!pushed_ok) {
                err = std::string("nkv_set_point_draw_ranges failed: ") + engine.last_error();
                return 0;
            }
            ++stats_.range_pushes;
        } else {
            ++stats_.range_pushes_skipped;
        }
        last_cut_complete_ = !incomplete;
        last_firsts_ = list.firsts;
        last_counts_ = list.counts;
        draw_ranges_valid_ = true;
        stats_.ranges_pushed = (merge_ranges_ && !merged_f_.empty()) ? merged_f_.size()
                                                                     : last_firsts_.size();
        stats_.cut_points = list.points;
        stats_.stand_in_ranges = list.stand_ins;
        stats_.stand_in_points = list.stand_in_points;
        if (list.stand_ins > 0) ++stats_.stand_in_frames;
        stats_.uncovered_cells = list.uncovered_nodes;
        stats_.uncovered_cells_total += list.uncovered_nodes;
        stats_.max_uncovered_cells = std::max(stats_.max_uncovered_cells, list.uncovered_nodes);
        stats_.cut_nodes = cut;
        return last_firsts_.size();
    }

    // Blanks the draw list without changing the selection - the "camera turned
    // away" end state, which must produce zero draw calls rather than a stale
    // list. Engine only.
    inline bool hide_all(VulkanEngine& engine) {
        draw_ranges_valid_ = false;
        stats_.ranges_pushed = 0;
        stats_.cut_points = 0;
        stats_.cut_nodes = 0;
        return engine.clear_point_draw_ranges();
    }

    // Number of directory nodes never uploaded by the lazy path. This is the
    // streaming saving, stated in the same units as the file.
    uint64_t nodes_never_uploaded() const {
        return tree_.nodes.size() > stats_.tiles_uploaded
                   ? tree_.nodes.size() - stats_.tiles_uploaded : 0;
    }

    const double* file_origin() const { return origin_; }

private:
    // TRUE WHEN THE RESIDENT CUT STILL DESCRIBES THIS CAMERA. Two independent
    // limits have to hold, and they are deliberately different quantities:
    //
    //   * MOTION, relative to the finest node in the cut: 0.01 * its size. A
    //     camera move shorter than a hundredth of the smallest cell on screen
    //     cannot change any node's screen-space error by a whole pixel at the
    //     default 90 px threshold, and it cannot push a cell across a frustum
    //     plane either: the cull test already grants each plane a slack that is a
    //     fixed fraction of the camera's own far clip (see LNTCamera.h), so a cell
    //     that is INSIDE a plane by more than the whole slack would have to move
    //     by more than that slack to become outside it. Scaling by the finest node
    //     rather than the coarsest is the conservative choice: a deep cut is the
    //     one whose visibility a small movement can actually change, and a coarse
    //     cut is the one the movement matters least for.
    //   * ROTATION about the eye: 0.05 degrees, the angle at which a cell at the
    //     far edge of a 60-degree field of view has moved by a pixel at 1080p.
    //
    // Everything that changes what a cut MEANS - the projection, the viewport, the
    // ortho/perspective switch, the origin shift, or an engine matrix replacing a
    // reconstructed one - invalidates it outright rather than being measured.
    //
    // A false "reuse" is a stale frame; a false "re-traverse" is a traversal of a
    // few hundred nodes. That asymmetry is why the limits are tight.
    bool can_reuse_selection(const LNTCamera& cam, bool* refused_for_data = nullptr) const {
        if (refused_for_data) *refused_for_data = false;
        if (!selection_valid_ || !cam.frustum_valid || !selection_camera_.frustum_valid)
            return false;
        const LNTCamera& prev = selection_camera_;
        if (cam.ortho != prev.ortho || cam.screen_height_px != prev.screen_height_px ||
            cam.aspect != prev.aspect || cam.fov_y_deg != prev.fov_y_deg ||
            cam.near_clip != prev.near_clip || cam.far_clip != prev.far_clip ||
            cam.ortho_half_height != prev.ortho_half_height ||
            cam.from_engine_mvp != prev.from_engine_mvp)
            return false;
        if (min_selected_node_size_ <= 0) return false;
        const double limit = 0.01 * min_selected_node_size_;
        auto moved = [&](const double a[3], const double b[3]) {
            const double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
            return dx * dx + dy * dy + dz * dz > limit * limit;
        };
        if (moved(cam.eye, prev.eye) || moved(cam.target, prev.target) ||
            moved(cam.origin, prev.origin)) return false;
        const double a[3] = {cam.target[0] - cam.eye[0], cam.target[1] - cam.eye[1],
                             cam.target[2] - cam.eye[2]};
        const double b[3] = {prev.target[0] - prev.eye[0], prev.target[1] - prev.eye[1],
                             prev.target[2] - prev.eye[2]};
        double dot = 0, aa = 0, bb = 0;
        for (int i = 0; i < 3; ++i) { dot += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
        if (aa <= 0 || bb <= 0) return false;
        if (dot / std::sqrt(aa * bb) <
            std::cos(0.05 * 3.14159265358979323846 / 180.0))
            return false;                       // the camera alone says rebuild

        // A payload ARRIVING can open a gate the resident cut was holding shut,
        // and it must open it while the camera is perfectly still: otherwise a
        // viewer that stops zooming stays at the coarse level the gate was
        // waiting on, for ever. So a load that happened since the cut was built
        // forces a rebuild - but it is a rebuild the CAMERA did not ask for, and
        // the caller is told that, because the two reasons mean different things
        // to everything downstream ("a still camera costs nothing" remains true,
        // "nothing re-ran" does not, and saying which is the honest report).
        if (params_.gate_split_on_residency && cache_ != nullptr &&
            !stats_.last_selection.gated_nodes.empty() &&
            cache_->loads() != selection_loads_) {
            if (refused_for_data) *refused_for_data = true;
            return false;
        }
        return true;
    }

    // RAII: every return path out of the engine overload - including the early
    // ones that report an error - records its main-thread cost.
    struct FrameTimerCall {
        LntStreamStats* s;
        std::chrono::steady_clock::time_point t0;
        explicit FrameTimerCall(LntStreamStats* stats)
            : s(stats), t0(std::chrono::steady_clock::now()) {}
        ~FrameTimerCall() {
            const auto t1 = std::chrono::steady_clock::now();
            const uint64_t ns = uint64_t(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
            s->main_ns = ns;
            if (ns > s->max_main_ns) s->max_main_ns = ns;
        }
    };

    // ---- the draw list's helpers -----------------------------------------

    // Appends one arena window. `covers_id` is the cut node the window
    // represents: equal to `owner_id` for a node's own tile, and the missing node
    // for a stand-in.
    inline void add_window(const LntTile& tile, uint64_t owner_id, uint64_t covers_id,
                           bool stand_in) {
        draw_list_.firsts.push_back(tile.first);
        draw_list_.counts.push_back(tile.count);
        draw_list_.ids.push_back(owner_id);
        draw_list_.covers.push_back(covers_id);
        draw_list_.points += tile.count;
        if (stand_in) {
            ++draw_list_.stand_ins;
            draw_list_.stand_in_points += tile.count;
        }
    }

    // Marks the node at `idx` for this frame's draw list; true when it was already
    // marked. A stamp generation rather than a per-frame clear: the array is the
    // size of the tree, and clearing it every frame would be O(nodes) for a list
    // that touches a few hundred.
    inline bool stamp_seen(uint32_t idx) {
        if (stamp_.size() < tree_.nodes.size()) stamp_.resize(tree_.nodes.size(), 0u);
        if (stamp_[idx] == stamp_gen_) return true;
        stamp_[idx] = stamp_gen_;
        return false;
    }

    // The nearest STRICT ancestor of `idx` whose tile is on the GPU, or nullptr.
    // Strict because the node itself is tested by the caller, and testing it
    // here as well would turn "this payload has not arrived" into "draw this
    // node's own range" - which is the bug being fixed.
    inline const LntTile* first_uploaded_ancestor(const LntScene& scene,
                                                 uint32_t idx) const {
        for (uint32_t cur = tree_.nodes[idx].parent; cur != kNoParent;
             cur = tree_.nodes[cur].parent) {
            const LntTile* t = scene.find_tile(tree_.nodes[cur].id);
            if (t != nullptr && t->count > 0 && t->uploaded) return t;
        }
        return nullptr;
    }

    LntNodeTree tree_;
    LNTCamera selection_camera_;      // the camera stats_.last_selection was built from
    // The cut that was built, kept so the NEXT rebuild can compare against it:
    // the coarsening band is a statement about two consecutive cuts, so the
    // traversal needs the previous one rather than a recomputation of it.
    LntLodSelection selection_prev_;
    // Cache loads observed when the cut was built; a difference means a payload
    // landed while the gate was shut.
    uint64_t selection_loads_ = 0;
    // A cut exists to compare the next one against, independent of selection_valid_
    // (which says whether the resident cut still answers THIS camera).
    bool has_previous_ = false;
    bool selection_valid_ = false;
    bool draw_ranges_valid_ = false;  // the engine holds last_firsts_/last_counts_
    // Size of the smallest node in the resident cut, i.e. the resolution the cut
    // actually achieved; the motion hysteresis is a fraction of it.
    double min_selected_node_size_ = 0.0;
    // The last (first, count) windows handed to the engine, kept so an unchanged
    // list can skip the call and an emptied list can clear exactly once.
    std::vector<uint32_t> last_firsts_, last_counts_;
    const std::vector<LNTNodeEntry>* entries_ = nullptr;
    LntCacheManager* cache_ = nullptr;
    LntLodParams params_{};
    double upload_budget_ms_ = 0.0;
    int base_levels_ = -1;
    double prefetch_ahead_ = 0.0;
    bool merge_ranges_ = false;
    bool last_cut_complete_ = false; // only a complete displayed cut can be held
    bool hold_incomplete_ = false;
    int stalled_holds_ = 0;
    std::vector<uint64_t> prefetch_ids_;
    std::unordered_set<uint64_t> issued_;    // ids this streamer asked the cache for
    uint64_t prefetch_requested_ = 0;
    std::vector<uint32_t> merged_f_, merged_c_;
    std::vector<uint64_t> base_ids_;
    double effective_sse_ = 0.0;
    uint64_t requests_cancelled_ = 0;
    // The resolved draw list and the per-frame stamp that keeps a shared stand-in
    // from being pushed once per cut node it covers.
    LntDrawList draw_list_;
    std::vector<uint32_t> stamp_;
    uint32_t stamp_gen_ = 0;
    bool ancestor_fallback_ = true;
    LntStreamStats stats_{};
    double origin_[3] = {0.0, 0.0, 0.0};
    uint64_t dataset_points_ = 0;
};

// Tool-facing report formatter. Declared here but DEFINED non-inline in
// src/LNTStreamer.cpp, so the single-command build (`g++ ... main.cpp -o main`)
// needs no second object file as long as main.cpp does not call it - the same
// arrangement as describe_file() in LNTReader.h / LNTReader.cpp.
std::string describe_streaming(const LntStreamer& streamer, const LntCacheManager& cache);

}  // namespace lnt

#endif  // LNT_STREAMER_H
