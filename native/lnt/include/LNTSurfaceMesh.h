// LNTSurfaceMesh.h - the shaded SURFACE of a streamed .lnt view: a triangulated, hillshaded
// terrain model built natively from the points the viewer has resident. No VTK, no Python.
//
//   points (per tile) --> Delaunay in plan --> triangles lifted to their real Z
//                     --> edge filter (drop triangles spanning gaps / voids)
//                     --> per-face normal --> hillshade N.L (azimuth, elevation, ambient)
//                     --> per-face colour (class palette | RGB | intensity ramp | elevation
//                         ramp) multiplied by the shade
//
// WHY PER TILE, WITH A HALO. A view is made of many octree tiles of different sizes.
// Triangulating each on its own leaves a gap along every tile border (each tile's hull stops
// at its last point), which shows as a grid of cracks. So a tile is triangulated TOGETHER
// with the points of its neighbours that lie within a few point-spacings of its border, and
// it KEEPS a triangle only if it is the OWNER: the lowest-numbered tile among the tiles the
// triangle's three vertices belong to. Every triangle has exactly one owner, so the tiles
// partition the surface - no gap, no overlap - and the rule needs no cell geometry, so it
// holds for tiles stacked on top of each other (an octree splits Z too) and for tiles of
// different sizes. The halo is wider than the longest edge the filter keeps, so a triangle's
// owner always has all three of its vertices in its point set.
//
// THE EDGE FILTER. A Delaunay triangulation also triangulates empty ground (a lake, a building
// footprint with no returns). A triangle whose longest plan edge exceeds
// `max_edge_factor` x the tile's MEDIAN longest edge is dropped, which leaves those voids open
// instead of bridging them with a sliver.
//
// The lighting matches gui/shading_display.py::_compute_shading (the same hillshade the rest of
// the application uses):  zenith = 90 - elevation;  azimuth maths angle = 360 - azimuth + 90;
//     L = (sin z cos a, sin z sin a, cos z);   shade = max(N.L, ambient), at most 1.

#ifndef LNT_SURFACE_MESH_H
#define LNT_SURFACE_MESH_H

#include <algorithm>
#include <atomic>
#include <array>
#include <bitset>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

#include "LNTDelaunay.h"
#include "LNTShadingModes.h"

namespace lnt {

struct SurfacePoint {
    double x, y, z;
    uint8_t cls, r, g, b, intensity;
};

struct SurfaceTile {
    double cmin[3], cmax[3];            // the tile's world cell
    std::vector<SurfacePoint> pts;
    // Optional zero-copy plan partition of the decoded leaf records. Each
    // visible source record appears exactly once in one tile's reference list.
    std::vector<const SurfacePoint*> point_refs;
    std::bitset<256> classes; // sampled classes, independent of palette visibility
    std::size_t point_count() const { return point_refs.empty() ? pts.size() : point_refs.size(); }
    const SurfacePoint& point(std::size_t i) const { return point_refs.empty() ? pts[i] : *point_refs[i]; }
};

enum class SurfaceColor : uint32_t { Class = 0, RGB = 1, Intensity = 2, Elevation = 3 };

enum class SurfaceMethod : uint32_t { Facet = 0, Barycentric = 1 };


struct SurfaceParams {
    SurfaceMethod method = SurfaceMethod::Facet;
    double azimuth_deg = 315.0;         // compass: where the light comes FROM
    double elevation_deg = 45.0;        // above the horizon
    double ambient = 0.25;              // shade floor, 0..1
    double max_edge_factor = 4.0;
    SurfaceColor color_source = SurfaceColor::Class;
    LNTColorRamp ramp = LNTColorRamp::Hue;
    double z_min = 0.0, z_max = 1.0;    // elevation ramp range (world Z)
    double i_min = 0.0, i_max = 255.0;  // intensity ramp range (stored 8-bit values)
    uint8_t class_rgb[256][3];          // class palette
    uint8_t class_visible[256];         // 0 hides a class: its points are not triangulated

    // ---- geometry parity with gui/shading_display.py::_compute_shading_geometry_backend ----
    // Every visible point is a mesh vertex (VTK's Slow / all-points policy): no point is
    // ever thinned out, and a point no kept triangle references is drawn as a point.
    // Light elevation for a SINGLE visible class (app.last_shade_angle): VTK lights that case
    // with the plain hillshade and bakes it into cell colours. <= 0 keeps elevation_deg.
    double single_elevation_deg = 0.0;
    // Filled by plan_surface() for the whole cut, then read by mesh_one_tile():
    double min_area = 0.0, min_aspect = 0.0;   // _numba_degenerate_filter (0 = off)
    double max_edge_abs = 0.0;          // >0 absolute limit, 0 median rule, <0 continuous
    // false when the engine lights and colours the mesh itself (compact crisp path): the
    // per-face colours and the mixed-face list are then not built, which saves host RAM.
    bool bake_colors = true;
    bool compact_crisp = false;
    double halo_fraction = 0.02;        // dense seams still get at least eight local point spacings

    SurfaceParams() {
        for (int i = 0; i < 256; ++i) {
            class_rgb[i][0] = class_rgb[i][1] = class_rgb[i][2] = 160;
            class_visible[i] = 1;
        }
    }
};

struct SurfaceMesh {
    std::vector<double> positions;      // world xyz, 3 per vertex
    std::vector<int32_t> faces;         // 3 vertex indices per triangle
    std::vector<double> coverage_positions; // own points not referenced by an emitted face
    std::vector<uint8_t> coverage_colors;
    uint64_t visible_points = 0;
    uint64_t mesh_referenced_points = 0;
    std::vector<uint8_t> vertex_colors; // RGB, baked vertex hillshade
    std::vector<uint8_t> colors;        // flat RGB per triangle (shade already applied)
    std::vector<uint32_t> blocks;       // optional: index count of each tile's run in `faces`
    // Shaded-classification upload inputs (see the engine's nkv_set_shaded_class_surface):
    // the display class of every vertex, one byte per vertex, and the global indices of the
    // faces whose three vertices show different display colours - exactly the crisp-hybrid
    // split baked by gui/shading_display.py::_collect_mixed_display_faces().
    std::vector<uint8_t> vertex_class;
    std::vector<int32_t> mixed_faces;
    uint64_t input_points = 0;
    uint64_t dropped_long_edges = 0;
    double build_ms = 0.0;
    uint64_t triangles() const { return faces.size() / 3; }
    uint64_t vertices() const { return positions.size() / 3; }
};

// The unit vector TOWARD the light for a compass azimuth / elevation (see the header).
inline void surface_light_vector(double azimuth_deg, double elevation_deg, double out[3]) {
    const double kDeg = 3.14159265358979323846 / 180.0;
    const double zenith = (90.0 - std::min(std::max(elevation_deg, 0.0), 90.0)) * kDeg;
    const double az = (360.0 - azimuth_deg + 90.0) * kDeg;
    out[0] = std::sin(zenith) * std::cos(az);
    out[1] = std::sin(zenith) * std::sin(az);
    out[2] = std::cos(zenith);
}

namespace detail {

// Fine-grained mesh timing: thread-summed nanoseconds per phase, reset and
// printed by lnt_view_api.cpp when NAKSHA_LNT_MESH_TIMING=1. Six atomic adds
// per tile is nothing next to the triangulation itself.
//   0 gather (own filter + occupancy + halo scan)  1 Delaunay
//   2 ownership/edge-limit scan (+ barycentric normals)
//   3 emit (filters, colours, vertices)            4 coverage fill
inline std::atomic<uint64_t> g_mesh_ns[5];
inline std::atomic<uint64_t> g_mesh_pts{0};
inline std::atomic<uint64_t> g_mesh_tris{0};
inline void mesh_timing_reset() {
    for (auto& c : g_mesh_ns) c.store(0, std::memory_order_relaxed);
    g_mesh_pts.store(0, std::memory_order_relaxed);
    g_mesh_tris.store(0, std::memory_order_relaxed);
}

struct TileMesh {
    std::vector<double> positions;
    std::vector<int32_t> faces;                 // local indices
    std::vector<double> coverage_positions; // own points not referenced by an emitted face
    std::vector<uint8_t> coverage_colors;
    uint64_t visible_points = 0;
    uint64_t mesh_referenced_points = 0;
    std::vector<uint32_t> coverage_point_indices;
    std::vector<uint8_t> local_referenced;
    std::vector<std::pair<uint32_t,uint32_t>> halo_references;
    std::vector<uint8_t> vertex_colors; // RGB, baked vertex hillshade
    std::vector<uint8_t> colors;
    std::vector<uint8_t> vertex_class;   // own-class byte per emitted vertex
    std::vector<int32_t> mixed_faces;    // LOCAL face indices whose 3 display colours differ
    uint64_t dropped = 0;
    // Surface LOD tiles only (collect_fringe): triangles of this tile's own Delaunay that
    // touch an own point but are OWNED by a lower-numbered neighbour, with that owner's
    // index. Drawn only where the owner is not, they close mixed-LOD seams. They never
    // count as references for coverage, so `faces` + coverage still account for every
    // own visible point on their own.
    bool collect_fringe = false;
    std::vector<int32_t> fringe_faces;   // local vertex indices, after `faces`' vertices
    std::vector<uint32_t> fringe_owner;  // one owner tile index per fringe triangle
    std::vector<const SurfacePoint*> vertex_src;   // collect_fringe: the record of each vertex
};

// The packed display colour of one class, matching gui/shading_display.py::_display_color_key_lut.
// Two faces with the same key at all three corners are "pure" and take the flat facet colour;
// otherwise they are "mixed" and are drawn with barycentric per-vertex colours.
inline uint32_t surface_class_key(const SurfaceParams& P, uint8_t cls) {
    return (uint32_t(P.class_rgb[cls][0]) << 16) |
           (uint32_t(P.class_rgb[cls][1]) << 8) |
            uint32_t(P.class_rgb[cls][2]);
}

inline bool xy_overlap(const double* amin, const double* amax, const double* bmin, const double* bmax) {
    return !(amax[0] < bmin[0] || bmax[0] < amin[0] || amax[1] < bmin[1] || bmax[1] < amin[1]);
}

// Reuse one X-interval index for the entire cut. A tile only tests cells whose
// X bounds can meet its halo, instead of scanning every leaf for every mesh.
struct TileXIndex {
    std::vector<std::size_t> order;
    std::vector<double> prefix_max;
    explicit TileXIndex(const std::vector<const SurfaceTile*>& tiles) {
        order.resize(tiles.size());
        for(std::size_t i=0;i<tiles.size();++i) order[i]=i;
        std::sort(order.begin(),order.end(),[&](std::size_t a,std::size_t b) {
            return tiles[a]->cmin[0]<tiles[b]->cmin[0];
        });
        prefix_max.resize(order.size());
        double max_x=-std::numeric_limits<double>::infinity();
        for(std::size_t i=0;i<order.size();++i) {
            max_x=std::max(max_x,tiles[order[i]]->cmax[0]);
            prefix_max[i]=max_x;
        }
    }
};

inline void mesh_one_tile(const std::vector<const SurfaceTile*>& tiles, std::size_t ti,
                          const TileXIndex& xindex, const SurfaceParams& P,
                          const double L[3], TileMesh& out) {
    using mesh_clk = std::chrono::steady_clock;
    mesh_clk::time_point t_prev = mesh_clk::now();
    auto mark = [](int bucket, mesh_clk::time_point& prev, const mesh_clk::time_point& now) {
        g_mesh_ns[bucket].fetch_add(uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            now - prev).count()), std::memory_order_relaxed);
        prev = now;
    };
    const SurfaceTile& T = *tiles[ti];
    // visible own points
    std::vector<const SurfacePoint*> pts;
    std::vector<uint32_t> owner_of;                  // the tile each point belongs to
    std::vector<uint32_t> point_index_of;
    pts.reserve(T.point_count() + 256);
    owner_of.reserve(T.point_count() + 256);
    point_index_of.reserve(T.point_count() + 256);
    for (std::size_t i=0;i<T.point_count();++i) {
        const auto& p=T.point(i);
        if (P.class_visible[p.cls]) {
            pts.push_back(&p); owner_of.push_back(uint32_t(ti));point_index_of.push_back(uint32_t(i));
        }
    }
    const std::size_t n_own = pts.size();
    out.visible_points = n_own;
    std::vector<uint8_t> referenced(n_own, 0);
    out.local_referenced.assign(T.point_count(),0);
    out.coverage_point_indices.reserve(n_own / 8 + 1);
    // Every own visible point must be represented. Untriangulatable/filtered
    // points are retained in a native GPU point pass, including duplicate XY.
    struct OnExit {
        std::function<void()> fn;
        ~OnExit() noexcept(false) { if(std::uncaught_exceptions()==0) fn(); }
    } coverage{[&]() {
        const auto t_cov = mesh_clk::now();
        for (std::size_t i=0; i<n_own; ++i) if (!referenced[i]) {
            out.coverage_point_indices.push_back(point_index_of[i]);
        }
        for(std::size_t i=0;i<n_own;++i)
            if(referenced[i]) out.local_referenced[point_index_of[i]]=1;
        // Record a neighbour point once, even if many emitted faces share it.
        for(std::size_t i=n_own;i<referenced.size();++i)
            if(referenced[i]) out.halo_references.emplace_back(owner_of[i],point_index_of[i]);
        mark(4, t_prev, mesh_clk::now());
    }};
    // One or two points in this cell can still form a valid surface with
    // neighbouring cells. Only an empty cell has no triangle to own.
    if (n_own == 0) return;

    // point spacing from the occupied part of the cell: a tile need not be full
    const int G = 24;
    std::vector<uint8_t> occ(G * G, 0);
    const double wx = std::max(T.cmax[0] - T.cmin[0], 1e-9), wy = std::max(T.cmax[1] - T.cmin[1], 1e-9);
    for (const SurfacePoint* p : pts) {
        const int gx = std::min(G - 1, std::max(0, int((p->x - T.cmin[0]) / wx * G)));
        const int gy = std::min(G - 1, std::max(0, int((p->y - T.cmin[1]) / wy * G)));
        occ[gy * G + gx] = 1;
    }
    int occupied = 0;
    for (uint8_t o : occ) occupied += o;
    const double spacing = std::sqrt(std::max(occupied, 1) * (wx / G) * (wy / G) / double(n_own));
    // wider than the longest edge the filter can keep (about 1.8 spacings x max_edge_factor)
    // With VTK's absolute edge limit the halo widens (up to half the tile) so a void-bridging
    // triangle still has all three vertices in its owner's point set.
    double halo = std::min(2.0 * P.max_edge_factor * spacing, 0.5 * std::min(wx, wy));
    if (P.max_edge_abs > 0.0)
        halo = std::max(halo, std::min(P.max_edge_abs, P.halo_fraction * std::min(wx, wy)));

    // halo: neighbours' points near this tile's border (any Z: stacked tiles share a column)
    const double emin[2] = {T.cmin[0] - halo, T.cmin[1] - halo};
    const double emax[2] = {T.cmax[0] + halo, T.cmax[1] + halo};
    const auto first=std::lower_bound(xindex.prefix_max.begin(),xindex.prefix_max.end(),emin[0]);
    for (std::size_t k=std::size_t(first-xindex.prefix_max.begin());k<xindex.order.size();++k) {
        const std::size_t j=xindex.order[k];
        if(tiles[j]->cmin[0]>emax[0]) break;
        if (j == ti) continue;
        const SurfaceTile& O = *tiles[j];
        if (!xy_overlap(emin, emax, O.cmin, O.cmax)) continue;
        for (std::size_t oi=0;oi<O.point_count();++oi) {
            const SurfacePoint& p=O.point(oi);
            if (!P.class_visible[p.cls]) continue;
            if (p.x >= emin[0] && p.x <= emax[0] && p.y >= emin[1] && p.y <= emax[1]) {
                pts.push_back(&p);
                owner_of.push_back(uint32_t(j));
                point_index_of.push_back(uint32_t(oi));
            }
        }
    }

    referenced.resize(pts.size(), 0);

    {
        const auto now = mesh_clk::now();
        mark(0, t_prev, now);
        g_mesh_pts.fetch_add(uint64_t(pts.size()), std::memory_order_relaxed);
    }
    const double ox = 0.5 * (T.cmin[0] + T.cmax[0]), oy = 0.5 * (T.cmin[1] + T.cmax[1]);
    // A 2.5D TIN has one vertex per plan position. Choose the top visible
    // return explicitly, instead of allowing Delaunay's duplicate suppression
    // to choose a lower return according to source/flight-line order.
    std::vector<std::size_t> order(pts.size());
    for (std::size_t i=0;i<order.size();++i) order[i]=i;
    std::sort(order.begin(),order.end(),[&](std::size_t a,std::size_t b) {
        if(pts[a]->x!=pts[b]->x) return pts[a]->x<pts[b]->x;
        if(pts[a]->y!=pts[b]->y) return pts[a]->y<pts[b]->y;
        if(pts[a]->z!=pts[b]->z) return pts[a]->z>pts[b]->z;
        return a<b;
    });
    std::vector<std::size_t> representatives;
    std::vector<double> xy;
    representatives.reserve(pts.size()); xy.reserve(pts.size()*2);
    for(auto i:order) {
        if(!representatives.empty()) {
            const auto* previous=pts[representatives.back()];
            if(previous->x==pts[i]->x && previous->y==pts[i]->y) continue;
        }
        representatives.push_back(i);
        xy.push_back(pts[i]->x-ox); xy.push_back(pts[i]->y-oy);
    }
    Delaunay2D tri(xy);
    auto tr = tri.triangles();
    for(auto& i:tr) i=representatives[i];
    const std::size_t nt = tr.size() / 3;
    {
        const auto now = mesh_clk::now();
        mark(1, t_prev, now);
        g_mesh_tris.fetch_add(uint64_t(nt), std::memory_order_relaxed);
    }
    if (nt == 0) return;

    // triangles this tile OWNS (lowest tile index among their vertices), with their longest plan edge
    std::vector<uint32_t> own;
    std::vector<double> longest2;
    std::vector<uint32_t> fringe, fringe_owner;
    std::vector<double> fringe_longest2;
    own.reserve(nt);
    longest2.reserve(nt);
    for (std::size_t t = 0; t < nt; ++t) {
        const SurfacePoint* a = pts[tr[3 * t]];
        const SurfacePoint* b = pts[tr[3 * t + 1]];
        const SurfacePoint* c = pts[tr[3 * t + 2]];
        const uint32_t owner = std::min({owner_of[tr[3*t]],owner_of[tr[3*t+1]],owner_of[tr[3*t+2]]});
        const double abx = a->x - b->x, aby = a->y - b->y;
        const double bcx = b->x - c->x, bcy = b->y - c->y;
        const double cax = c->x - a->x, cay = c->y - a->y;
        const double m2 = std::max({abx*abx+aby*aby, bcx*bcx+bcy*bcy, cax*cax+cay*cay});
        if (owner != uint32_t(ti)) {
            if (out.collect_fringe && (owner_of[tr[3*t]] == uint32_t(ti) ||
                                       owner_of[tr[3*t+1]] == uint32_t(ti) ||
                                       owner_of[tr[3*t+2]] == uint32_t(ti))) {
                fringe.push_back(uint32_t(t));
                fringe_owner.push_back(owner);
                fringe_longest2.push_back(m2);
            }
            continue;
        }
        own.push_back(uint32_t(t));
        longest2.push_back(m2);
    }
    if (own.empty() && (fringe.empty() || P.max_edge_abs == 0.0)) return;
    double med=0;
    if(P.max_edge_abs == 0.0) {
        std::vector<double> sorted = longest2;
        std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
        med = std::sqrt(sorted[sorted.size() / 2]);
    }
    // VTK keeps every triangle up to an ABSOLUTE plan edge (10 % of the cut's extent with
    // several classes, 20 % with one); the legacy native rule is relative to the tile median.
    // Capped at the halo: a longer kept edge could miss its owner's point set (a seam).
    const double limit = P.max_edge_abs < 0.0 ? std::numeric_limits<double>::infinity()
                       : P.max_edge_abs > 0.0 ? std::min(P.max_edge_abs, halo)
                                              : P.max_edge_factor * med;
    const double limit2 = limit * limit;

    // Cross products are twice-area weighted normals. Include every kept halo
    // triangle before ownership filtering, so shared border vertices see neighbours.
    std::vector<std::array<double, 3>> normals;
    if (P.method == SurfaceMethod::Barycentric) {
        normals.resize(pts.size(), {0.0, 0.0, 0.0});
        for (std::size_t t = 0; t < nt; ++t) {
            const auto ia = tr[3*t], ib = tr[3*t+1], ic = tr[3*t+2];
            const auto *a = pts[ia], *b = pts[ib], *c = pts[ic];
            const double abx = a->x-b->x, aby = a->y-b->y;
            const double bcx = b->x-c->x, bcy = b->y-c->y;
            const double cax = c->x-a->x, cay = c->y-a->y;
            if (std::max({abx*abx+aby*aby, bcx*bcx+bcy*bcy, cax*cax+cay*cay}) > limit2) continue;
            const double ux=b->x-a->x, uy=b->y-a->y, uz=b->z-a->z;
            const double vx=c->x-a->x, vy=c->y-a->y, vz=c->z-a->z;
            double nx=uy*vz-uz*vy, ny=uz*vx-ux*vz, nz=ux*vy-uy*vx;
            if (nx*nx+ny*ny+nz*nz <= 1e-24) continue;
            if (nz < 0) { nx=-nx; ny=-ny; nz=-nz; }
            for (auto i : {ia,ib,ic}) {
                normals[i][0]+=nx; normals[i][1]+=ny; normals[i][2]+=nz;
            }
        }
    }
    {
        const auto now = mesh_clk::now();
        mark(2, t_prev, now);
    }
    std::vector<int32_t> remap(pts.size(), -1);
    // Ownership has already bounded the output. Reserve its maximum once to
    // avoid growing and copying multi-gigabyte streams across all the tiles.
    const std::size_t vertex_bound=std::min(pts.size(),own.size()*3);
    out.positions.reserve(vertex_bound*3);
    out.vertex_class.reserve(vertex_bound);
    out.faces.reserve(own.size()*3);
    if(P.bake_colors) out.colors.reserve(own.size()*3);
    if(P.method==SurfaceMethod::Barycentric) out.vertex_colors.reserve(vertex_bound*3);
    auto vertex = [&](std::size_t pi) {
        if (remap[pi] < 0) {
            remap[pi] = int32_t(out.positions.size() / 3);
            if (out.collect_fringe) out.vertex_src.push_back(pts[pi]);
            out.positions.push_back(pts[pi]->x);
            out.positions.push_back(pts[pi]->y);
            out.positions.push_back(pts[pi]->z);
            out.vertex_class.push_back(pts[pi]->cls);
            if (P.method == SurfaceMethod::Barycentric) {
                const auto& n = normals[pi];
                const double len = std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
                const double shade = std::min(1.0, std::max(P.ambient,
                    len > 1e-12 ? (n[0]*L[0]+n[1]*L[1]+n[2]*L[2])/len : L[2]));
                const auto& p = *pts[pi];
                double base[3];
                if (P.color_source == SurfaceColor::Class) {
                    for (int k=0;k<3;++k) base[k]=P.class_rgb[p.cls][k];
                } else if (P.color_source == SurfaceColor::RGB) {
                    base[0]=p.r; base[1]=p.g; base[2]=p.b;
                } else {
                    const double u = P.color_source == SurfaceColor::Elevation
                        ? (p.z-P.z_min)/std::max(P.z_max-P.z_min,1e-9)
                        : (p.intensity-P.i_min)/std::max(P.i_max-P.i_min,1e-9);
                    float rgb[3]; evaluate_ramp(P.ramp, float(u), rgb);
                    for (int k=0;k<3;++k) base[k]=rgb[k]*255.0;
                }
                for (int k=0;k<3;++k)
                    out.vertex_colors.push_back(uint8_t(std::min(255.0,std::max(0.0,base[k]*shade+0.5))));
            }
        }
        return remap[pi];
    };

    for (std::size_t k = 0; k < own.size(); ++k) {
        if (longest2[k] > limit2) { ++out.dropped; continue; }
        const std::size_t t = own[k];
        const std::size_t ia = tr[3 * t], ib = tr[3 * t + 1], ic = tr[3 * t + 2];
        const SurfacePoint *a = pts[ia], *b = pts[ib], *c = pts[ic];
        if (P.min_area > 0.0 || P.min_aspect > 0.0) {      // _numba_degenerate_filter
            const double area = 0.5 * std::abs((b->x - a->x) * (c->y - a->y) -
                                               (b->y - a->y) * (c->x - a->x));
            if (!(area > P.min_area && area / (longest2[k] + 1e-10) > P.min_aspect)) {
                ++out.dropped; continue;
            }
        }
        if(!P.bake_colors && P.min_area>=1e-12) {
            // The passed plan-area filter already proves the 3D normal is
            // nonzero. The compact GPU path computes lighting and colours;
            // do not repeat that work for every face on the CPU.
            for(auto i:{ia,ib,ic}) referenced[i]=1;
            out.faces.push_back(vertex(ia));
            out.faces.push_back(vertex(ib));
            out.faces.push_back(vertex(ic));
            continue;
        }
        const double ux = b->x - a->x, uy = b->y - a->y, uz = b->z - a->z;
        const double vx = c->x - a->x, vy = c->y - a->y, vz = c->z - a->z;
        double nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
        const double nl = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (!(nl > 1e-12)) continue;                     // a sliver with no area
        if (nz < 0.0) { nx = -nx; ny = -ny; nz = -nz; }  // the surface faces up
        const double ndotl = (nx * L[0] + ny * L[1] + nz * L[2]) / nl;
        const double shade = std::min(1.0, std::max(ndotl, P.ambient));

        double base[3];
        switch (P.color_source) {
            case SurfaceColor::RGB:
                base[0] = (a->r + b->r + c->r) / 3.0;
                base[1] = (a->g + b->g + c->g) / 3.0;
                base[2] = (a->b + b->b + c->b) / 3.0;
                break;
            case SurfaceColor::Intensity: {
                const double iv = (a->intensity + b->intensity + c->intensity) / 3.0;
                const double u = (iv - P.i_min) / std::max(P.i_max - P.i_min, 1e-9);
                float rgb[3];
                evaluate_ramp(P.ramp, float(u), rgb);
                for (int i = 0; i < 3; ++i) base[i] = rgb[i] * 255.0;
                break;
            }
            case SurfaceColor::Elevation: {
                const double zc = (a->z + b->z + c->z) / 3.0;
                const double u = (zc - P.z_min) / std::max(P.z_max - P.z_min, 1e-9);
                float rgb[3];
                evaluate_ramp(P.ramp, float(u), rgb);
                for (int i = 0; i < 3; ++i) base[i] = rgb[i] * 255.0;
                break;
            }
            default: {                                   // class: the majority class of the corners
                uint8_t k0 = a->cls;
                if (b->cls == c->cls) k0 = b->cls;
                for (int i = 0; i < 3; ++i) base[i] = P.class_rgb[k0][i];
                break;
            }
        }
        for (auto i : {ia,ib,ic}) {
            referenced[i]=1;
        }
        // Crisp-hybrid split: a face whose three vertices share one display colour is flat
        // (facet); a face whose displayed colours differ is a mixed face and is drawn with
        // barycentric per-vertex colours. Recorded here, in the mesher, because only the
        // mesher knows which faces survived the long-edge filter.
        if (P.color_source == SurfaceColor::Class && P.bake_colors) {
            const uint32_t k0 = surface_class_key(P, a->cls);
            const uint32_t k1 = surface_class_key(P, b->cls);
            const uint32_t k2 = surface_class_key(P, c->cls);
            if (k0 != k1 || k0 != k2)
                out.mixed_faces.push_back(int32_t(out.faces.size() / 3));
        }
        out.faces.push_back(vertex(ia));
        out.faces.push_back(vertex(ib));
        out.faces.push_back(vertex(ic));
        // Class colours are TRUNCATED like VTK's np.clip(lut * shade).astype(np.uint8).
        const double bias = P.color_source == SurfaceColor::Class ? 0.0 : 0.5;
        if (P.bake_colors)
        for (int i = 0; i < 3; ++i)
            out.colors.push_back(uint8_t(std::min(255.0, std::max(0.0, base[i] * shade + bias))));
    }
    // Seam fringe: the same edge and degenerate filters as owned faces. Vertices are
    // shared with the owned faces; `referenced` is deliberately not touched.
    for (std::size_t k = 0; k < fringe.size(); ++k) {
        if (fringe_longest2[k] > limit2) continue;
        const std::size_t t = fringe[k];
        const std::size_t ia = tr[3 * t], ib = tr[3 * t + 1], ic = tr[3 * t + 2];
        const SurfacePoint *a = pts[ia], *b = pts[ib], *c = pts[ic];
        if (P.min_area > 0.0 || P.min_aspect > 0.0) {
            const double area = 0.5 * std::abs((b->x - a->x) * (c->y - a->y) -
                                               (b->y - a->y) * (c->x - a->x));
            if (!(area > P.min_area && area / (fringe_longest2[k] + 1e-10) > P.min_aspect)) continue;
        }
        out.fringe_faces.push_back(vertex(ia));
        out.fringe_faces.push_back(vertex(ib));
        out.fringe_faces.push_back(vertex(ic));
        out.fringe_owner.push_back(fringe_owner[k]);
    }
    {
        const auto now = mesh_clk::now();
        mark(3, t_prev, now);
    }
}

inline void reconcile_coverage(std::vector<TileMesh>& parts,
                               const std::vector<const SurfaceTile*>& tiles,
                               const SurfaceParams& P, const double L[3]) {
    for(const auto& part:parts)
        for(const auto& ref:part.halo_references)
            if(ref.first<parts.size() && ref.second<parts[ref.first].local_referenced.size())
                parts[ref.first].local_referenced[ref.second]=1;
    const double shade=std::min(1.0,std::max(P.ambient,L[2]));
    for(std::size_t ti=0;ti<parts.size();++ti) {
        auto& part=parts[ti];
        part.mesh_referenced_points=std::count(part.local_referenced.begin(),part.local_referenced.end(),uint8_t(1));
        for(const auto index:part.coverage_point_indices) {
            if(part.local_referenced[index]) continue;
            const auto& q=tiles[ti]->point(index);
            part.coverage_positions.insert(part.coverage_positions.end(),{q.x,q.y,q.z});
            double base[3];
            if(P.color_source==SurfaceColor::Class) {
                for(int k=0;k<3;++k) base[k]=P.class_rgb[q.cls][k];
            } else if(P.color_source==SurfaceColor::RGB) {
                base[0]=q.r;base[1]=q.g;base[2]=q.b;
            } else {
                const double u=P.color_source==SurfaceColor::Elevation
                    ? (q.z-P.z_min)/std::max(P.z_max-P.z_min,1e-9)
                    : (q.intensity-P.i_min)/std::max(P.i_max-P.i_min,1e-9);
                float rgb[3];evaluate_ramp(P.ramp,float(u),rgb);
                for(int k=0;k<3;++k) base[k]=rgb[k]*255;
            }
            for(int k=0;k<3;++k)
                part.coverage_colors.push_back(uint8_t(std::min(255.0,std::max(0.0,base[k]*shade+0.5))));
        }
        part.coverage_point_indices.clear();
        part.local_referenced.clear();
        part.halo_references.clear();
    }
}

}  // namespace detail

// ---- whole-cut filter parameters (gui/shading_display.py, all-points policy) -----------
// Every visible point is kept (VTK's Slow mode: nothing is thinned). From the cut:
//   spacing = sqrt(extent_x * extent_y / n_visible)
//   degenerate filter: numeric only (zero-area triangles), as VTK's Slow
//   edge filter: plan edge <= 10 % of the cut's extent (20 % with one visible class)
// A point left unreferenced by these filters is still drawn (coverage pass), never skipped.
// Host bytes per point of a full-detail Shading cut (decoded record + mesh vertex + its two
// triangles + bookkeeping). Used for the RAM check before a build starts.
constexpr uint64_t kSurfaceBytesPerPoint = 96;

struct SurfacePlan {
    uint64_t visible = 0;
    double spacing = 0.0, extent = 0.0;
    double xy_min[2]{},xy_max[2]{};
};

inline void plan_surface(const std::vector<const SurfaceTile*>& tiles, SurfaceParams& P,
                         unsigned visible_classes, SurfacePlan& plan) {
    plan = SurfacePlan();
    double mn[2] = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    double mx[2] = {-mn[0], -mn[1]};
    for (const auto* t : tiles)
        for (std::size_t i=0;i<t->point_count();++i) {
            const auto& p=t->point(i);
            if (!P.class_visible[p.cls]) continue;
            ++plan.visible;
            mn[0] = std::min(mn[0], p.x); mx[0] = std::max(mx[0], p.x);
            mn[1] = std::min(mn[1], p.y); mx[1] = std::max(mx[1], p.y);
        }
    if (plan.visible < 3) return;
    for(int a=0;a<2;++a) {plan.xy_min[a]=mn[a];plan.xy_max[a]=mx[a];}
    const double xr = mx[0] - mn[0], yr = mx[1] - mn[1];
    plan.spacing = std::sqrt((xr * yr) / double(plan.visible));
    plan.extent = std::max(xr, yr);
    P.min_area = 1e-12;
    P.min_aspect = 1e-10;
    P.max_edge_abs = plan.extent * (visible_classes == 1 ? 0.20 : 0.10);
    if (const char* h = std::getenv("NAKSHA_LNT_HALO_FRACTION")) {
        char* end=nullptr;
        const double fraction=std::strtod(h,&end);
        if(end!=h && *end=='\0' && std::isfinite(fraction) && fraction>=0.01 && fraction<=0.5)
            P.halo_fraction=fraction;
    }
}

// The file's octree partitions Z too. Repartition the SAME visible records in
// XY for a plan triangulation; keep references so no coordinate or class is
// copied, rounded, deduplicated, or sampled. All source leaves remain alive.
inline void partition_surface_plan(const std::vector<const SurfaceTile*>& source,
                                   const SurfaceParams& P, const SurfacePlan& plan,
                                   std::vector<SurfaceTile>& output,
                                   std::size_t target_points=50000) {
    if(!target_points) throw std::invalid_argument("plan tile point target must be positive");
    const double wx=std::max(plan.xy_max[0]-plan.xy_min[0],1e-9);
    const double wy=std::max(plan.xy_max[1]-plan.xy_min[1],1e-9);
    const double cells=std::max(1.0,double(plan.visible)/double(target_points));
    const unsigned nx=unsigned(std::clamp(std::ceil(std::sqrt(cells*wx/wy)),1.0,256.0));
    const unsigned ny=unsigned(std::clamp(std::ceil(cells/nx),1.0,256.0));
    output.resize(nx*ny);
    auto cell_of=[&](const SurfacePoint& p) {
        const unsigned x=unsigned(std::clamp((p.x-plan.xy_min[0])/wx*nx,0.0,double(nx-1)));
        const unsigned y=unsigned(std::clamp((p.y-plan.xy_min[1])/wy*ny,0.0,double(ny-1)));
        return y*nx+x;
    };
    std::vector<std::size_t> counts(output.size());
    for(const auto* t:source) for(std::size_t i=0;i<t->point_count();++i) {
        const auto& p=t->point(i);
        if(P.class_visible[p.cls]) ++counts[cell_of(p)];
    }
    for(unsigned y=0;y<ny;++y) for(unsigned x=0;x<nx;++x) {
        auto& t=output[y*nx+x];
        t.cmin[0]=plan.xy_min[0]+wx*x/nx;t.cmax[0]=plan.xy_min[0]+wx*(x+1)/nx;
        t.cmin[1]=plan.xy_min[1]+wy*y/ny;t.cmax[1]=plan.xy_min[1]+wy*(y+1)/ny;
        t.cmin[2]=t.cmax[2]=0;
        t.point_refs.reserve(counts[y*nx+x]);
    }
    uint64_t assigned=0;
    for(const auto* t:source) for(std::size_t i=0;i<t->point_count();++i) {
        const auto& p=t->point(i);
        if(!P.class_visible[p.cls]) continue;
        auto& cell=output[cell_of(p)];
        cell.point_refs.push_back(&p);cell.classes.set(p.cls);
        ++assigned;
    }
    if(assigned!=plan.visible) throw std::runtime_error("plan partition lost visible records");
}

// Builds the surface of every tile and concatenates it. Tiles are independent, so they run on
// `threads` workers (0 = the machine's concurrency).
inline SurfaceMesh build_surface_mesh(const std::vector<SurfaceTile>& tiles, const SurfaceParams& params,
                                      unsigned threads = 0) {
    const auto t0 = std::chrono::steady_clock::now();
    SurfaceMesh mesh;
    double L[3];
    surface_light_vector(params.azimuth_deg, params.elevation_deg, L);
    std::vector<detail::TileMesh> parts(tiles.size());
    std::vector<const SurfaceTile*> tile_refs;
    tile_refs.reserve(tiles.size());
    for (const auto& tile : tiles) tile_refs.push_back(&tile);
    const detail::TileXIndex xindex(tile_refs);
    if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
    threads = std::min<unsigned>(threads, unsigned(std::max<std::size_t>(tiles.size(), 1)));
    std::atomic<std::size_t> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr mesh_error;
    std::mutex mesh_error_mu;
    auto worker = [&]() {
        try {
            for (;;) {
                if(failed.load()) return;
                const std::size_t i = next.fetch_add(1);
                if (i >= tiles.size()) return;
                detail::mesh_one_tile(tile_refs, i, xindex, params, L, parts[i]);
            }
        } catch(...) {
            std::lock_guard<std::mutex> lock(mesh_error_mu);
            if(!mesh_error) mesh_error=std::current_exception();
            failed.store(true);
        }
    };
    std::vector<std::thread> pool;
    for (unsigned i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (std::thread& t : pool) t.join();
    if(mesh_error) std::rethrow_exception(mesh_error);
    detail::reconcile_coverage(parts, tile_refs, params, L);

    std::size_t verts = 0, tris = 0;
    for (const detail::TileMesh& p : parts) { verts += p.positions.size() / 3; tris += p.faces.size() / 3; }
    mesh.positions.reserve(verts * 3);
    mesh.faces.reserve(tris * 3);
    mesh.colors.reserve(tris * 3);
    mesh.blocks.reserve(parts.size());
    for (std::size_t i = 0; i < parts.size(); ++i) {
        const detail::TileMesh& p = parts[i];
        const int32_t base = int32_t(mesh.positions.size() / 3);
        const int32_t face_base = int32_t(mesh.faces.size() / 3);
        // The engine's indexed uploads take one index count per tile run. Leaving this empty
        // is not harmless: the coverage-carrying upload refuses a mesh that has unmeshed
        // points but no block table, so a two-point coverage gap silently dropped the WHOLE
        // surface. Each run is this tile's own triangle count, in `faces` order (empty tiles
        // contribute no run, matching the streamed builder).
        if (!p.faces.empty()) mesh.blocks.push_back(uint32_t(p.faces.size()));
        mesh.positions.insert(mesh.positions.end(), p.positions.begin(), p.positions.end());
        for (int32_t f : p.faces) mesh.faces.push_back(base + f);
        mesh.colors.insert(mesh.colors.end(), p.colors.begin(), p.colors.end());
        mesh.vertex_colors.insert(mesh.vertex_colors.end(), p.vertex_colors.begin(), p.vertex_colors.end());
        mesh.vertex_class.insert(mesh.vertex_class.end(), p.vertex_class.begin(), p.vertex_class.end());
        for (int32_t m : p.mixed_faces) mesh.mixed_faces.push_back(face_base + m);
        mesh.coverage_positions.insert(mesh.coverage_positions.end(),p.coverage_positions.begin(),p.coverage_positions.end());
        mesh.coverage_colors.insert(mesh.coverage_colors.end(),p.coverage_colors.begin(),p.coverage_colors.end());
        mesh.visible_points += p.visible_points;
        mesh.mesh_referenced_points += p.mesh_referenced_points;
        mesh.dropped_long_edges += p.dropped;
        mesh.input_points += tiles[i].point_count();
    }
    mesh.build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return mesh;
}

}  // namespace lnt

#endif  // LNT_SURFACE_MESH_H
