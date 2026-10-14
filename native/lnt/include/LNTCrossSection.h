// LNTCrossSection.h - cross-section cutting geometry for the .lnt pipeline.
//
// Pure value types and math: no allocation, no engine, no file access. A cross
// section is a VOLUME of space the viewer keeps; everything outside it is
// discarded on the GPU (see VulkanEngine::set_clip_volume) and its octree nodes
// are never loaded (see LntStreamer::query_nodes_in_corridor).
//
//   Plane   A x + B y + C z + D = 0, normal (A,B,C) pointing INTO the kept side:
//           a point is kept when A x + B y + C z + D >= 0.
//   ClipBox centre, half-extents and an orthonormal frame (the corridor of a road
//           or rail profile). It is expressed as six inward-facing planes, which is
//           what the GPU receives; the engine takes at most four at a time, so a
//           section corridor is sent as its two side planes plus the two end caps
//           and the vertical extent is handled by the node corridor query.
//
// Coordinates are WORLD coordinates (float64) everywhere in this file. The engine
// converts to its float32 render space when it uploads the planes.

#ifndef LNT_CROSS_SECTION_H
#define LNT_CROSS_SECTION_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "LNTReader.h"   // BoundingBox

namespace lnt {

static constexpr uint32_t kMaxClipPlanes = 4;

// ---- Plane ----------------------------------------------------------------
struct Plane {
    double a = 0.0, b = 0.0, c = 1.0, d = 0.0;

    Plane() = default;
    Plane(double a_, double b_, double c_, double d_) : a(a_), b(b_), c(c_), d(d_) {}

    // Plane through `point` whose kept side is the side `normal` points to.
    // `normal` need not be unit length; a zero normal gives the degenerate plane
    // (0,0,0,0) which keeps everything.
    static Plane from_point_normal(const double point[3], const double normal[3]) {
        const double len = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
                                     normal[2] * normal[2]);
        if (!(len > 0.0)) return Plane(0, 0, 0, 0);
        const double n0 = normal[0] / len, n1 = normal[1] / len, n2 = normal[2] / len;
        return Plane(n0, n1, n2, -(n0 * point[0] + n1 * point[1] + n2 * point[2]));
    }

    // Horizontal plane z = z0, keeping z >= z0 (up = true) or z <= z0.
    static Plane horizontal(double z0, bool keep_above) {
        return keep_above ? Plane(0, 0, 1, -z0) : Plane(0, 0, -1, z0);
    }

    // Signed distance (exact when the normal is unit length): >= 0 is kept.
    double distance(double x, double y, double z) const { return a * x + b * y + c * z + d; }
    bool keeps(double x, double y, double z) const { return distance(x, y, z) >= 0.0; }

    bool degenerate() const { return a == 0.0 && b == 0.0 && c == 0.0; }
};

// ---- plane vs AABB --------------------------------------------------------
// Where a box lies relative to a plane. The test projects the box onto the plane
// normal: its centre distance and its "radius" along the normal,
//     r = hx*|A| + hy*|B| + hz*|C|
// give the exact signed-distance interval [dist - r, dist + r] over the box.
enum class PlaneSide : uint8_t {
    Inside = 0,     // whole box on the kept side      (dist - r >= 0)
    Outside = 1,    // whole box on the discarded side (dist + r <  0)
    Straddles = 2   // the plane cuts the box
};

inline PlaneSide classify_aabb(const Plane& p, const BoundingBox& box) {
    if (p.degenerate()) return PlaneSide::Inside;
    const double cx = box.mid(0), cy = box.mid(1), cz = box.mid(2);
    const double hx = 0.5 * box.extent(0), hy = 0.5 * box.extent(1), hz = 0.5 * box.extent(2);
    const double dist = p.distance(cx, cy, cz);
    const double r = hx * std::fabs(p.a) + hy * std::fabs(p.b) + hz * std::fabs(p.c);
    if (dist - r >= 0.0) return PlaneSide::Inside;
    if (dist + r < 0.0) return PlaneSide::Outside;
    return PlaneSide::Straddles;
}

// True when ANY part of the box can lie on the kept side. This is the cull test:
// a node that returns false has no point the plane keeps.
inline bool plane_aabb_intersects(const Plane& p, const BoundingBox& box) {
    return classify_aabb(p, box) != PlaneSide::Outside;
}

// ---- ClipBox ---------------------------------------------------------------
struct ClipBox {
    double center[3] = {0, 0, 0};
    double half[3] = {0.5, 0.5, 0.5};            // half-extent along each frame axis
    // Orthonormal frame, ROW-major: axis[i] is the unit vector the i-th half-extent
    // measures along. Identity = an axis-aligned box.
    double axis[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    // The profile corridor of a section line P1 -> P2: `along` runs from P1 to P2,
    // `across` is the in-plane normal (half_width each side), `up` is world Z with
    // the given vertical half-extent about `cz`. This is exactly the volume the
    // cross-section tool draws as its rectangle.
    static ClipBox corridor(const double p1[2], const double p2[2], double half_width,
                            double cz, double half_height) {
        ClipBox b;
        double dx = p2[0] - p1[0], dy = p2[1] - p1[1];
        const double len = std::sqrt(dx * dx + dy * dy);
        if (!(len > 0.0)) { dx = 1.0; dy = 0.0; }
        else { dx /= len; dy /= len; }
        b.center[0] = 0.5 * (p1[0] + p2[0]);
        b.center[1] = 0.5 * (p1[1] + p2[1]);
        b.center[2] = cz;
        b.half[0] = 0.5 * len;
        b.half[1] = half_width;
        b.half[2] = half_height;
        b.axis[0][0] = dx;  b.axis[0][1] = dy;  b.axis[0][2] = 0;
        b.axis[1][0] = -dy; b.axis[1][1] = dx;  b.axis[1][2] = 0;
        b.axis[2][0] = 0;   b.axis[2][1] = 0;   b.axis[2][2] = 1;
        return b;
    }

    bool contains(double x, double y, double z) const {
        const double rel[3] = {x - center[0], y - center[1], z - center[2]};
        for (int i = 0; i < 3; ++i) {
            const double t = rel[0] * axis[i][0] + rel[1] * axis[i][1] + rel[2] * axis[i][2];
            if (std::fabs(t) > half[i]) return false;
        }
        return true;
    }

    // The six inward-facing planes: +axis end then -axis end for i = 0, 1, 2.
    std::vector<Plane> planes() const {
        std::vector<Plane> out;
        out.reserve(6);
        for (int i = 0; i < 3; ++i) {
            for (int s = 0; s < 2; ++s) {
                const double sign = (s == 0) ? -1.0 : 1.0;      // normal points inward
                const double n[3] = {-sign * axis[i][0], -sign * axis[i][1], -sign * axis[i][2]};
                const double pt[3] = {center[0] + sign * half[i] * axis[i][0],
                                      center[1] + sign * half[i] * axis[i][1],
                                      center[2] + sign * half[i] * axis[i][2]};
                out.push_back(Plane::from_point_normal(pt, n));
            }
        }
        return out;
    }

    // World-space AABB of the oriented box: what the node corridor query culls with.
    BoundingBox aabb() const {
        BoundingBox b;
        for (int k = 0; k < 3; ++k) {
            const double ext = half[0] * std::fabs(axis[0][k]) + half[1] * std::fabs(axis[1][k]) +
                               half[2] * std::fabs(axis[2][k]);
            b.min[k] = center[k] - ext;
            b.max[k] = center[k] + ext;
        }
        return b;
    }

    // Exact oriented-box vs AABB overlap (separating-axis test): the box's three
    // frame axes plus the AABB's three world axes. The 9 edge-cross-product axes are
    // omitted on purpose: the result is then a SUPERSET (never misses a node that
    // holds a kept point), which is the direction a cull must err in.
    bool intersects(const BoundingBox& o) const {
        if (o.is_empty()) return false;
        const double oc[3] = {o.mid(0), o.mid(1), o.mid(2)};
        const double oh[3] = {0.5 * o.extent(0), 0.5 * o.extent(1), 0.5 * o.extent(2)};
        const double d[3] = {oc[0] - center[0], oc[1] - center[1], oc[2] - center[2]};
        for (int i = 0; i < 3; ++i) {              // this box's axes
            const double dist = std::fabs(d[0] * axis[i][0] + d[1] * axis[i][1] + d[2] * axis[i][2]);
            const double r = oh[0] * std::fabs(axis[i][0]) + oh[1] * std::fabs(axis[i][1]) +
                             oh[2] * std::fabs(axis[i][2]);
            if (dist > half[i] + r) return false;
        }
        const BoundingBox mine = aabb();            // the AABB's axes
        return mine.intersects(o);
    }
};

}  // namespace lnt

#endif  // LNT_CROSS_SECTION_H
