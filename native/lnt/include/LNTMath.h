// LNTMath.h - quantisation and 64-bit Morton key math.
//
// Zero dependencies, header-only, all functions constexpr-friendly inline.
//
// The two properties this file exists to guarantee:
//
//   1. NO BIT OVERFLOW. 21 bits/axis interleaves into 63 of the 64 available
//      bits. The expansion is done with mask-and-shift, not a per-bit loop, so
//      there is no intermediate state that can exceed 64 bits. The earlier
//      Python reference had two real bugs in this area: an interleave that
//      OR-ed whole values instead of single bits (collapsing a million points
//      onto ~800 keys), and a quantiser scaled by (2**bits - 1) instead of
//      2**bits, which made the top level a zero-width sliver reachable only at
//      exactly v == max. Both are fixed; neither pattern appears here.
//
//   2. NO ZERO-WIDTH SLIVER AT THE TOP. `quantize_to_21bit` scales by the
//      number of BINS (2**bits), not the highest index (2**bits - 1), so every
//      one of the 2**bits levels is reachable and every bin has identical width,
//      and the top shard/octant is as populated as any other.
//
//      `quantize_to_16bit` uses the OTHER convention on purpose, and the two are
//      not interchangeable. It has to invert `node_local_to_world()`, which the
//      format defines as `lo + rel * (2*half) / 65535`, so the forward map must
//      round to the nearest of 65535 INTERVALS, not floor into 65536 BINS. That
//      pairing is what bounds a decoded point's error by HALF a step instead of
//      a full one: floor(u*65536) inverted by /65535 leaves a worst case of
//      65535/65536 = 0.99998 of a step, so the payload was delivering half the
//      precision it was built for. See quantize_to_16bit below.

#ifndef LNT_MATH_H
#define LNT_MATH_H

#include <cstdint>

#include "LNTFormat.h"

namespace lnt {

// ========================================================================
// 21-bit bit expansion
// ========================================================================
// Spread the low 21 bits of v so that bit i lands at bit 3*i, leaving two
// zero bits between each. Classic mask-and-shift doubling: 5 steps, no loop,
// no branch. Intermediate masks are chosen so the value never occupies more
// than the 63 bits it ends in.
inline uint64_t expand_bits_21(uint64_t v) {
    v &= 0x1FFFFFULL;                            // keep exactly 21 bits
    v = (v | (v << 32)) & 0x1F00000000FFFFULL;
    v = (v | (v << 16)) & 0x1F0000FF0000FFULL;
    v = (v | (v << 8)) & 0x100F00F00F00F00FULL;
    v = (v | (v << 4)) & 0x10C30C30C30C30C3ULL;
    v = (v | (v << 2)) & 0x1249249249249249ULL;
    return v;
}

// Exact inverse of expand_bits_21: gather bits at 3*i back down to bit i.
inline uint64_t compress_bits_21(uint64_t v) {
    v &= 0x1249249249249249ULL;
    v = (v ^ (v >> 2)) & 0x10C30C30C30C30C3ULL;
    v = (v ^ (v >> 4)) & 0x100F00F00F00F00FULL;
    v = (v ^ (v >> 8)) & 0x1F0000FF0000FFULL;
    v = (v ^ (v >> 16)) & 0x1F00000000FFFFULL;
    v = (v ^ (v >> 32)) & 0x1FFFFFULL;
    return v;
}

// ========================================================================
// 64-bit Morton (Z-order) key
// ========================================================================
// Bit assignment: x -> positions 0,3,6,...  y -> 1,4,7,...  z -> 2,5,8,...
// so the LOWEST octant bit of a 3-bit group is X, then Y, then Z. Everything
// downstream (octant split, child cell selection) uses that same convention.
inline uint64_t encode_morton_64(uint32_t x, uint32_t y, uint32_t z) {
    // Mask rather than reject: the per-axis width is FIXED at 21 by the format,
    // and quantize_to_21bit() can only ever return 0..kAxisMax, so this mask is
    // a hard guarantee against overflow rather than a lossy truncation. A
    // caller that hands in a raw value above 2^21-1 is out of contract, and
    // masking keeps the key well-formed instead of wrapping silently.
    const uint32_t mx = x & uint32_t(kAxisMax);
    const uint32_t my = y & uint32_t(kAxisMax);
    const uint32_t mz = z & uint32_t(kAxisMax);
    return expand_bits_21(mx) | (expand_bits_21(my) << 1) | (expand_bits_21(mz) << 2);
}

inline void decode_morton_64(uint64_t code, uint32_t& x, uint32_t& y, uint32_t& z) {
    x = uint32_t(compress_bits_21(code));
    y = uint32_t(compress_bits_21(code >> 1));
    z = uint32_t(compress_bits_21(code >> 2));
}

// The octant index that selects this node's child at `level`, for level in
// 0..kAxisBits-1. Returns 0..7 with bit0 = X, bit1 = Y, bit2 = Z.
//
// BEWARE THE WIDTH. A 21-bit-per-axis key occupies 63 bits, NOT 64, so the top
// bit of the code is always 0 and the octant bits live at 3*(20-level), not at
// 64 - 3*(level+1). Using the 64-bit form shifts every level by one bit group:
// it reads three always-zero bits at level 0 and reports octant 0 for every
// point in the outer half of the cube - a silent, total loss of the X split.
inline uint32_t octant_at(uint64_t code, int level) {
    const int shift = 3 * (kAxisBits - 1 - level);
    if (shift < 0 || level >= kAxisBits) return 0;
    return uint32_t((code >> shift) & 0x7u);
}

// The Morton prefix of a node at `level`: its `level` octant groups, packed
// into the low 3*level bits. Level 0 (the root) is the empty prefix, 0.
static constexpr int kMaxTreeLevel = kAxisBits;   // 21

inline uint64_t morton_prefix(uint64_t code, int level) {
    if (level <= 0) return 0;
    if (level >= kAxisBits) return code & 0x7FFFFFFFFFFFFFFFULL;  // 63 bits
    return code >> (3 * (kAxisBits - level));
}

// A node's ID: its Morton prefix with the LEVEL encoded above it.
//
// Why the tag is required. A bare prefix is NOT unique across levels: the root
// has the empty prefix 0, and the child in octant 0 has prefix 0 too, so both
// would be named 0 - two different nodes claiming the same id in one table.
// Tagging with 1 << (3*level) puts each level in its own bit-length range
// (level 0 -> 1, level 1 -> 8..15, level 2 -> 64..127, ...), so ids cannot
// collide, the level is recoverable from the id alone, and the child/parent
// relation still holds as `child_id >> 3 == parent_id`.
inline uint64_t node_id_at(uint64_t code, int level) {
    if (level < 0) level = 0;
    if (level > kAxisBits) level = kAxisBits;
    return (1ull << (3 * level)) | morton_prefix(code, level);
}

// Recover the level from a node id: the position of its tag bit, over 3.
inline int level_of_node_id(uint64_t id) {
    int top = 0;
    while (top < 63 && (id >> (top + 1)) != 0) ++top;
    return top / 3;
}

// ========================================================================
// Quantisation
// ========================================================================
// Two conventions, each the exact inverse of its own dequantiser - see the file
// header. `quantize_to_21bit` floors into 2**bits bins; `quantize_to_16bit`
// rounds onto the 65535 intervals that node_local_to_world() divides by.
//
// Comparison against NaN: `!(a > b)` is used deliberately instead of `a <= b`
// so that a NaN coordinate lands in the safe branch instead of propagating.
// This is not decoration. `std::clamp(NaN, 0.0, 1.0)` returns NaN (neither
// `NaN < 0` nor `1 < NaN` is true), so a clamp-based version would reach
// `static_cast<uint16_t>(NaN)`, which is undefined behaviour rather than a
// clamped value.

inline uint32_t quantize_to_21bit(double val, double min_val, double max_val) {
    if (!(max_val > min_val)) return 0;          // degenerate or NaN range
    const double u = (val - min_val) / (max_val - min_val);
    if (!(u > 0.0)) return 0;                    // clamps NaN and negatives
    if (u >= 1.0) return uint32_t(kAxisMax);     // clamp BEFORE multiplying
    const double q = u * double(kAxisLevels);    // 2**bits bins, not 2**bits-1
    if (!(q > 0.0)) return 0;
    if (q > double(kAxisMax)) return uint32_t(kAxisMax);
    return uint32_t(q);
}

// MID-BIN ROUNDING, and the exact inverse of node_local_to_world().
//
// `rel` indexes 65535 intervals, so rounding `u * 65535` to the nearest integer
// makes the decode error at most half an interval:
//
//     |q - u*65535| <= 0.5   =>   |world_decoded - world| <= 0.5 * step
//
// The previous floor(u * 65536) form left a worst case of 65535/65536 of a step
// because the forward and inverse conventions disagreed, which cost the payload
// half its precision.
//
// What this changes about the bin structure, stated so it is not a surprise: the
// interior levels are full width and the TWO boundary levels (0 and 65535) are
// half width, because a rounded bin is centred on its own grid point. That is the
// standard shape of a round-to-nearest quantiser and it is exactly what buys the
// factor of two - it is not a sliver. A sliver is a bin reachable at ONE input
// value, which is what the v1 defect produced; these are each half a step of
// real input range, and level 65535 is reached by every u >= 1 - 0.5/65535.
//
// The endpoint u == 1 is folded into the top level rather than clipped away, so
// `val == max_val` still round-trips exactly.
inline uint16_t quantize_to_16bit(double val, double min_val, double max_val) {
    if (!(max_val > min_val)) return 0;              // degenerate or NaN range
    const double u = (val - min_val) / (max_val - min_val);
    if (!(u > 0.0)) return 0;                        // clamps NaN and negatives
    if (u >= 1.0) return uint16_t(kLocalMax);        // endpoint folded in
    const double q = u * double(kLocalMax) + 0.5;    // 65535 intervals, rounded
    if (!(q > 0.0)) return 0;
    if (q >= double(kLocalMax)) return uint16_t(kLocalMax);
    return uint16_t(q);
}

// ---- inverses, for verification and for the reader's geometry path -----
// Each takes the SAME type its forward quantiser returns: uint32_t for the
// 21-bit key (a key component is naturally 32-bit) and uint16_t for the local
// payload, which is what LNTPoint actually stores. Matching signatures are part
// of the point of this pair - the defect this file carried was exactly a
// disagreement between the forward and inverse conventions.
inline double dequantize_from_21bit(uint32_t q, double min_val, double max_val) {
    return min_val + (max_val - min_val) * double(q) / double(kAxisMax);
}

inline double dequantize_from_16bit(uint16_t q, double min_val, double max_val) {
    return min_val + (max_val - min_val) * double(q) / double(kLocalMax);
}

// The per-point reconstruction contract:
//
//     world = (bounds_min + center - half_size) + rel * (2 * half_size) / 65535
//
// `origin` is the header's bounds_min, and `center` is a float32 offset from
// it - NOT an absolute world coordinate. That distinction is load-bearing:
//
//   * An ABSOLUTE float32 centre at a UTM-like magnitude is quantised to the
//     float32 ulp of that magnitude. At 6.8e5 m the ulp is 0.0625 m, so a node
//     cell smaller than ~6 cm cannot be described at all, and the rounding
//     lands directly in the decoded point. Measured on a 100k build, the
//     anchor error pushed the leaf reconstruction error to 1.55x the payload's
//     own 16-bit step (0.0142 m against a 0.00916 m step) - the 16-bit payload
//     was being wasted, and the error came from the anchor, not the data.
//   * RELATIVE to bounds_min, the offset is at most the cube's side (1200 m in
//     that build), where the float32 ulp is 1.22e-4 m - two orders of magnitude
//     below the 9.16e-3 m cell step, so the payload is once again the limit.
//
// The 52-byte LNTNodeEntry layout is unchanged; only the meaning of `center`
// is pinned down.
inline double node_local_to_world(uint32_t rel, float center_rel, float half_size,
                                  double origin) {
    const double lo = origin + double(center_rel) - double(half_size);
    return lo + 2.0 * double(half_size) * double(rel) / double(kLocalMax);
}

// Quantisation step of a node cell: the smallest resolvable displacement. This
// is the honest error bound for a decoded point, and it is what the verifier
// asserts against rather than an arbitrary tolerance.
inline double node_cell_step(float half_size) {
    return 2.0 * double(half_size) / double(kLocalMax);
}

// float32 ulp at a given magnitude, for checking that an anchor is precise
// enough to be worth storing. Returns 0 for non-finite input.
inline double float_ulp(double magnitude) {
    if (!(magnitude > 0.0) || !(magnitude < 1e300)) return 0.0;
    int e = 0;
    double m = magnitude;
    while (m >= 2.0) { m *= 0.5; ++e; }
    while (m < 1.0) { m *= 2.0; --e; }
    // 24 significant bits: ulp = 2^(exponent - 23)
    double ulp = 1.0;
    const int shift = e - 23;
    for (int i = 0; i < (shift > 0 ? shift : -shift); ++i) ulp *= 2.0;
    return shift > 0 ? ulp : 1.0 / ulp;
}

}  // namespace lnt

#endif  // LNT_MATH_H
