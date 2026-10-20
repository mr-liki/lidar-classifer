// LNTTypes.h - plain value types shared by the scanner and the builder.
//
// No allocation, no inheritance, no virtuals: these are PODs the scanner can
// fill from several threads and the reducer can merge without locking.

#ifndef LNT_TYPES_H
#define LNT_TYPES_H

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace lnt {

// ---- RawPoint: the converter's input contract ---------------------------
// XYZ authoritative in float64 (survey coordinates are large: a UTM northing
// of 3.5e6 m), with 8-bit colour and intensity. Input readers are expected to
// hand the builder an array of these, so the builder never depends on any file
// format library.
struct RawPoint {
    double   x;
    double   y;
    double   z;
    uint8_t  r;
    uint8_t  g;
    uint8_t  b;
    uint8_t  intensity;
    // ASPRS classification code (0..255). Sits in what used to be tail padding, so
    // sizeof(RawPoint) is unchanged (32 on LP64/LLP64).
    uint8_t  classification;
    uint16_t source_id = 0;
};

// ---- Bounds3d ----------------------------------------------------------
struct Bounds3d {
    double min[3];
    double max[3];

    void set_empty() {
        for (int i = 0; i < 3; ++i) {
            min[i] = +HUGE_VAL;
            max[i] = -HUGE_VAL;
        }
    }

    bool is_empty() const { return !(max[0] >= min[0]); }

    void expand(double x, double y, double z) {
        const double v[3] = {x, y, z};
        for (int i = 0; i < 3; ++i) {
            if (v[i] < min[i]) min[i] = v[i];
            if (v[i] > max[i]) max[i] = v[i];
        }
    }

    void expand(const Bounds3d& o) {
        if (o.is_empty()) return;
        for (int i = 0; i < 3; ++i) {
            if (o.min[i] < min[i]) min[i] = o.min[i];
            if (o.max[i] > max[i]) max[i] = o.max[i];
        }
    }

    double extent(int axis) const { return max[axis] - min[axis]; }

    double max_extent() const {
        double e = extent(0);
        if (extent(1) > e) e = extent(1);
        if (extent(2) > e) e = extent(2);
        return e;
    }

    void center(double out[3]) const {
        for (int i = 0; i < 3; ++i) out[i] = 0.5 * (min[i] + max[i]);
    }

    // Force a PERFECT CUBE with equal sides on X, Y and Z.
    //
    // Why it is required: the octree divides a node into eight children by
    // halving the cell on every axis. If the envelope is not cubic, the X, Y
    // and Z children are not the same size, so a node's own `half_size` (one
    // float, shared by all three axes in LNTNodeEntry) cannot describe it and
    // the 16-bit local quantisation of LNTPoint would silently mean a different
    // physical step per axis.
    //
    // The expansion is SYMMETRIC about the existing centre: the survey stays
    // centred and no already-valid data can fall outside. A flat site (a 120 m
    // Z range against a 1200 m X range) therefore gets padded in Z, which costs
    // bit budget but is the only way to keep one cube.
    void make_cube() {
        if (is_empty()) return;
        const double side = max_extent();
        if (!(side > 0.0)) {                 // zero-size or NaN input
            const double c[3] = {min[0], min[1], min[2]};
            for (int i = 0; i < 3; ++i) { min[i] = c[i]; max[i] = c[i]; }
            return;
        }
        const double half = 0.5 * side;
        double c[3];
        center(c);
        for (int i = 0; i < 3; ++i) {
            min[i] = c[i] - half;
            max[i] = c[i] + half;
        }
    }

    // Tolerance-based check, for tests. A cube built by make_cube() is exact
    // when `side` is representable, but adding/subtracting `half` from a centre
    // as large as 3.5e6 m rounds in the last bits, so an exact == test would be
    // flaky. 1e-6 m is far below any survey precision.
    bool is_cube(double tol = 1e-6) const {
        const double ex = extent(0), ey = extent(1), ez = extent(2);
        return std::fabs(ex - ey) <= tol && std::fabs(ex - ez) <= tol;
    }
};

// ---- point count of a source, for progress reporting -------------------
struct ScanResult {
    Bounds3d bounds;
    uint64_t points = 0;
    double   seconds = 0.0;
    unsigned threads = 0;
};

}  // namespace lnt

#endif  // LNT_TYPES_H
