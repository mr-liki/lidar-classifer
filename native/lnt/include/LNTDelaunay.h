// LNTDelaunay.h - 2D Delaunay triangulation (sweep-hull, O(n log n)).
//
// The native shaded surface triangulates the points of the view in plan (x, y) and lifts
// the triangles to their real z. This is the sweep-hull algorithm popularised by
// Delaunator: pick a seed triangle near the centre, sort the other points by distance from
// its circumcentre, add them one by one to a convex hull held in a hash for O(1) lookup of
// the visible edge, and flip illegal edges as they appear.
//
// Coordinates should be LOCAL (subtract a nearby origin first): survey coordinates around
// 6.8e5 m leave a double only ~1e-10 m of precision, and the in-circle test squares them.
//
// Output: `triangles` holds 3 point indices per triangle, all CLOCKWISE in a y-up plane
// (negative signed area; a consumer that needs a normal must not assume otherwise); `halfedges`
// pairs each triangle edge with the opposite one (kInvalid on the hull). Duplicate points
// (within machine epsilon of the previous one in sweep order) are skipped, so their index
// never appears. Fewer than 3 distinct points, or all collinear, gives an empty result.

#ifndef LNT_DELAUNAY_H
#define LNT_DELAUNAY_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace lnt {

class Delaunay2D {
public:
    // Tile-local indices fit in 32 bits. Halve the triangulation scratch without
    // changing coordinates, predicates, sweep order or the points processed.
    using Index = uint32_t;
    static constexpr Index kInvalid = std::numeric_limits<Index>::max();

    // coords = x0, y0, x1, y1, ...
    explicit Delaunay2D(const std::vector<double>& coords, bool fast_sort=true)
        : coords_(coords) { build(fast_sort); }

    const std::vector<Index>& triangles() const { return triangles_; }
    const std::vector<Index>& halfedges() const { return halfedges_; }
    std::size_t triangle_count() const { return triangles_.size() / 3; }

private:
    static constexpr double kEps = std::numeric_limits<double>::epsilon();

    const std::vector<double>& coords_;
    std::vector<Index> triangles_, halfedges_;
    std::vector<Index> hull_prev_, hull_next_, hull_tri_, hash_, edge_stack_;
    Index hull_start_ = 0, hash_size_ = 0;
    double cx_ = 0.0, cy_ = 0.0;

    static Index fast_mod(Index i, Index c) { return i >= c ? i % c : i; }
    static double dist2(double ax, double ay, double bx, double by) {
        const double dx = ax - bx, dy = ay - by;
        return dx * dx + dy * dy;
    }
    static double circumradius2(double ax, double ay, double bx, double by, double cx, double cy) {
        const double dx = bx - ax, dy = by - ay, ex = cx - ax, ey = cy - ay;
        const double bl = dx * dx + dy * dy, cl = ex * ex + ey * ey, d = dx * ey - dy * ex;
        if ((bl > 0.0 || bl < 0.0) && (cl > 0.0 || cl < 0.0) && (d > 0.0 || d < 0.0)) {
            const double x = (ey * bl - dy * cl) * 0.5 / d, y = (dx * cl - ex * bl) * 0.5 / d;
            return x * x + y * y;
        }
        return std::numeric_limits<double>::max();
    }
    // true when p, q, r turn CLOCKWISE (negative area) - the sweep's "orient" predicate
    static bool cw(double px, double py, double qx, double qy, double rx, double ry) {
        return (qy - py) * (rx - qx) - (qx - px) * (ry - qy) < 0.0;
    }
    static void circumcentre(double ax, double ay, double bx, double by, double cx, double cy,
                             double& ox, double& oy) {
        const double dx = bx - ax, dy = by - ay, ex = cx - ax, ey = cy - ay;
        const double bl = dx * dx + dy * dy, cl = ex * ex + ey * ey, d = dx * ey - dy * ex;
        ox = ax + (ey * bl - dy * cl) * 0.5 / d;
        oy = ay + (dx * cl - ex * bl) * 0.5 / d;
    }
    // true when p lies strictly inside the circle through a, b, c (a, b, c counter-clockwise)
    static bool in_circle(double ax, double ay, double bx, double by, double cx, double cy,
                          double px, double py) {
        const double dx = ax - px, dy = ay - py, ex = bx - px, ey = by - py, fx = cx - px, fy = cy - py;
        const double ap = dx * dx + dy * dy, bp = ex * ex + ey * ey, cp = fx * fx + fy * fy;
        return (dx * (ey * cp - bp * fy) - dy * (ex * cp - bp * fx) + ap * (ex * fy - ey * fx)) < 0.0;
    }

    Index hash_key(double x, double y) const {
        const double dx = x - cx_, dy = y - cy_;
        const double p = dx / (std::fabs(dx) + std::fabs(dy));
        const double a = (dy > 0.0 ? 3.0 - p : 1.0 + p) / 4.0;
        return fast_mod(static_cast<Index>(std::floor(a * double(hash_size_))), hash_size_);
    }

    void link(Index a, Index b) {
        halfedges_[a] = b;
        if (b != kInvalid) halfedges_[b] = a;
    }
    Index add_triangle(Index i0, Index i1, Index i2, Index a, Index b, Index c) {
        const Index t = Index(triangles_.size());
        triangles_.push_back(i0);
        triangles_.push_back(i1);
        triangles_.push_back(i2);
        halfedges_.push_back(kInvalid);
        halfedges_.push_back(kInvalid);
        halfedges_.push_back(kInvalid);
        link(t, a);
        link(t + 1, b);
        link(t + 2, c);
        return t;
    }

    Index legalize(Index a) {
        Index i = 0, ar = 0;
        edge_stack_.clear();
        for (;;) {
            const Index b = halfedges_[a];
            const Index a0 = 3 * (a / 3);
            ar = a0 + (a + 2) % 3;
            if (b == kInvalid) {
                if (i > 0) { a = edge_stack_[--i]; continue; }
                break;
            }
            const Index b0 = 3 * (b / 3);
            const Index al = a0 + (a + 1) % 3;
            const Index bl = b0 + (b + 2) % 3;
            const Index p0 = triangles_[ar], pr = triangles_[a], pl = triangles_[al],
                              p1 = triangles_[bl];
            const bool illegal = in_circle(coords_[2 * p0], coords_[2 * p0 + 1], coords_[2 * pr],
                                           coords_[2 * pr + 1], coords_[2 * pl], coords_[2 * pl + 1],
                                           coords_[2 * p1], coords_[2 * p1 + 1]);
            if (illegal) {
                triangles_[a] = p1;
                triangles_[b] = p0;
                const Index hbl = halfedges_[bl];
                if (hbl == kInvalid) {          // the flipped edge was on the hull: fix its owner
                    Index e = hull_start_;
                    do {
                        if (hull_tri_[e] == bl) { hull_tri_[e] = a; break; }
                        e = hull_prev_[e];
                    } while (e != hull_start_);
                }
                link(a, hbl);
                link(b, halfedges_[ar]);
                link(ar, bl);
                const Index br = b0 + (b + 1) % 3;
                if (edge_stack_.size() <= i) edge_stack_.resize(i + 16);
                edge_stack_[i++] = br;
            } else {
                if (i > 0) { a = edge_stack_[--i]; continue; }
                break;
            }
        }
        return ar;
    }

    void build(bool fast_sort) {
        const std::size_t n = coords_.size() >> 1;
        if (n < 3) return;
        // At most 2*n triangles and six edge indices per input point. Reject
        // an oversized tile explicitly rather than wrapping or omitting points.
        if (n > (uint64_t(kInvalid) - 1) / 6)
            throw std::length_error("Delaunay tile exceeds 32-bit edge capacity");
        double min_x = std::numeric_limits<double>::max(), min_y = min_x;
        double max_x = std::numeric_limits<double>::lowest(), max_y = max_x;
        std::vector<Index> ids(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double x = coords_[2 * i], y = coords_[2 * i + 1];
            min_x = std::min(min_x, x); min_y = std::min(min_y, y);
            max_x = std::max(max_x, x); max_y = std::max(max_y, y);
            ids[i] = i;
        }
        const double mx = (min_x + max_x) / 2.0, my = (min_y + max_y) / 2.0;

        // seed triangle: the point nearest the centre, its nearest neighbour, and the third
        // point that gives the smallest circumcircle
        double min_dist = std::numeric_limits<double>::max();
        Index i0 = kInvalid, i1 = kInvalid, i2 = kInvalid;
        for (std::size_t i = 0; i < n; ++i) {
            const double d = dist2(mx, my, coords_[2 * i], coords_[2 * i + 1]);
            if (d < min_dist) { i0 = i; min_dist = d; }
        }
        const double i0x = coords_[2 * i0], i0y = coords_[2 * i0 + 1];
        min_dist = std::numeric_limits<double>::max();
        for (std::size_t i = 0; i < n; ++i) {
            if (i == i0) continue;
            const double d = dist2(i0x, i0y, coords_[2 * i], coords_[2 * i + 1]);
            if (d < min_dist && d > 0.0) { i1 = i; min_dist = d; }
        }
        if (i1 == kInvalid) return;                       // every point is the same point
        double i1x = coords_[2 * i1], i1y = coords_[2 * i1 + 1];
        double min_radius = std::numeric_limits<double>::max();
        for (std::size_t i = 0; i < n; ++i) {
            if (i == i0 || i == i1) continue;
            const double r = circumradius2(i0x, i0y, i1x, i1y, coords_[2 * i], coords_[2 * i + 1]);
            if (r < min_radius) { i2 = i; min_radius = r; }
        }
        if (!(min_radius < std::numeric_limits<double>::max())) return;   // all collinear
        double i2x = coords_[2 * i2], i2y = coords_[2 * i2 + 1];
        if (cw(i0x, i0y, i1x, i1y, i2x, i2y)) {
            std::swap(i1, i2);
            std::swap(i1x, i2x);
            std::swap(i1y, i2y);
        }
        circumcentre(i0x, i0y, i1x, i1y, i2x, i2y, cx_, cy_);

        // sweep order: distance from the seed circumcentre
        const double ccx = cx_, ccy = cy_;
        {
        std::vector<uint64_t> distance_keys(n);
        for(std::size_t i=0;i<n;++i) {
            const double distance=dist2(coords_[2*i],coords_[2*i+1],ccx,ccy);
            if(!std::isfinite(distance)) throw std::runtime_error("non-finite Delaunay sweep distance");
            std::memcpy(&distance_keys[i],&distance,sizeof(distance));
        }
        auto xy_less=[&](Index i,Index j) {
            const double diff2 = coords_[2 * i] - coords_[2 * j];
            if (diff2 > 0.0 || diff2 < 0.0) return diff2 < 0.0;
            const double diff3=coords_[2*i+1]-coords_[2*j+1];
            if(diff3>0.0 || diff3<0.0) return diff3<0.0;
            return i<j; // stable identity for duplicate XY records
        };
        if(!fast_sort || n<4096) {
            std::sort(ids.begin(),ids.end(),[&](Index i,Index j) {
                return distance_keys[i]!=distance_keys[j]
                    ? distance_keys[i]<distance_keys[j] : xy_less(i,j);
            });
        } else {
            // Nonnegative IEEE double bits have the same order as their values.
            // Eight sequential byte passes replace random O(n log n) comparisons.
            std::vector<Index> scratch(n);
            for(unsigned shift=0;shift<64;shift+=8) {
                std::array<Index,256> offsets{};
                for(const auto i:ids) ++offsets[(distance_keys[i]>>shift)&255];
                Index total=0;
                for(auto& count:offsets) {Index next=total+count;count=total;total=next;}
                for(const auto i:ids) scratch[offsets[(distance_keys[i]>>shift)&255]++]=i;
                ids.swap(scratch);
            }
            // Preserve the reference XY tie-break, including quantized grids.
            for(std::size_t first=0;first<n;) {
                std::size_t last=first+1;
                while(last<n && distance_keys[ids[first]]==distance_keys[ids[last]]) ++last;
                if(last-first>1) std::sort(ids.begin()+first,ids.begin()+last,xy_less);
                first=last;
            }
        }
        } // release sort scratch before allocating hull and triangle arrays

        hash_size_ = static_cast<Index>(std::llround(std::ceil(std::sqrt(double(n)))));
        hash_.assign(hash_size_, kInvalid);
        hull_next_.assign(n, 0);
        hull_prev_.assign(n, 0);
        hull_tri_.assign(n, 0);
        hull_start_ = i0;
        hull_next_[i0] = hull_prev_[i2] = i1;
        hull_next_[i1] = hull_prev_[i0] = i2;
        hull_next_[i2] = hull_prev_[i1] = i0;
        hull_tri_[i0] = 0; hull_tri_[i1] = 1; hull_tri_[i2] = 2;
        hash_[hash_key(i0x, i0y)] = i0;
        hash_[hash_key(i1x, i1y)] = i1;
        hash_[hash_key(i2x, i2y)] = i2;
        triangles_.reserve(3 * (2 * n));
        halfedges_.reserve(3 * (2 * n));
        add_triangle(i0, i1, i2, kInvalid, kInvalid, kInvalid);

        double xp = std::numeric_limits<double>::quiet_NaN(), yp = xp;
        for (std::size_t k = 0; k < n; ++k) {
            const Index i = ids[k];
            const double x = coords_[2 * i], y = coords_[2 * i + 1];
            if (k > 0 && std::fabs(x - xp) <= kEps && std::fabs(y - yp) <= kEps) continue;
            xp = x; yp = y;
            if (i == i0 || i == i1 || i == i2) continue;

            // the hull edge this point can see
            Index start = 0;
            const Index key = hash_key(x, y);
            for (Index j = 0; j < hash_size_; ++j) {
                start = hash_[fast_mod(key + j, hash_size_)];
                if (start != kInvalid && start != hull_next_[start]) break;
            }
            start = hull_prev_[start];
            Index e = start, q;
            while (q = hull_next_[e],
                   !cw(x, y, coords_[2 * e], coords_[2 * e + 1], coords_[2 * q], coords_[2 * q + 1])) {
                e = q;
                if (e == start) { e = kInvalid; break; }
            }
            if (e == kInvalid) continue;                  // a near-duplicate

            Index t = add_triangle(e, i, hull_next_[e], kInvalid, kInvalid, hull_tri_[e]);
            hull_tri_[i] = legalize(t + 2);
            hull_tri_[e] = t;

            Index next = hull_next_[e];
            while (q = hull_next_[next],
                   cw(x, y, coords_[2 * next], coords_[2 * next + 1], coords_[2 * q], coords_[2 * q + 1])) {
                t = add_triangle(next, i, q, hull_tri_[i], kInvalid, hull_tri_[next]);
                hull_tri_[i] = legalize(t + 2);
                hull_next_[next] = next;                  // marks it as removed from the hull
                next = q;
            }
            if (e == start) {
                while (q = hull_prev_[e],
                       cw(x, y, coords_[2 * q], coords_[2 * q + 1], coords_[2 * e], coords_[2 * e + 1])) {
                    t = add_triangle(q, i, e, kInvalid, hull_tri_[e], hull_tri_[q]);
                    legalize(t + 2);
                    hull_tri_[q] = t;
                    hull_next_[e] = e;
                    e = q;
                }
            }
            hull_prev_[i] = e;
            hull_start_ = e;
            hull_prev_[next] = i;
            hull_next_[e] = i;
            hull_next_[i] = next;
            hash_[hash_key(x, y)] = i;
            hash_[hash_key(coords_[2 * e], coords_[2 * e + 1])] = e;
        }
    }
};

}  // namespace lnt

#endif  // LNT_DELAUNAY_H
