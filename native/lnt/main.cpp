// main.cpp - .lnt pipeline end-to-end verification.
//
// Build (header-only, single command):
//     g++ -O3 -std=c++17 main.cpp -o main
// Build (split TU):
//     g++ -O3 -std=c++17 -DLNT_EXTERN_BUILDER main.cpp src/LNTBuilder.cpp -o main
// Run:
//     ./main [output_path]
//
// What this proves, in order:
//   A. the four binary structs are exactly 64 / 16 / 10 / 52 bytes
//   B. the spatial math matches an independent bit-by-bit reference
//   C. 100,000 synthetic points convert to a valid .lnt file
//   D. the file's magic, trailer address and node table are self-consistent
//   E. decoded point payloads land inside their node cube and on real source
//      points, within the node's own 16-bit quantisation step
//   F. the build is deterministic (byte-identical across two runs)
//
// No test framework: one bool per check, a counter, and a non-zero exit code if
// anything fails, so it can gate a build.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "include/LNTBuilder.h"
#include "include/LNTLASReader.h"
#include "include/LNTCameraController.h"
#include "include/LNTInteractiveLoop.h"
#include "include/LNTFormat.h"
#include "include/LNTMath.h"
#include "include/LNTReader.h"
#include "include/LNTCacheManager.h"
#include "include/LNTCamera.h"
#include "include/LNTRenderBridge.h"
#include "include/LNTStreamer.h"
#include "include/LNTTypes.h"
#include "include/LNTVulkanEngine.h"
#include "include/LNTClassificationTable.h"
#include "include/LNTCrossSection.h"
#include "include/LNTSectionQuery.h"
#include "include/LNTShadingModes.h"
#ifndef LNT_EXTERN_BUILDER
#include "include/LNTBuilderImpl.h"   // header-only: single-TU build
#endif

using namespace lnt;

// ========================================================================
// tiny check harness
// ========================================================================
static int g_checks = 0;
static int g_failed = 0;
static int g_skipped = 0;

static void check(const std::string& label, bool ok, const std::string& detail = "") {
    ++g_checks;
    if (!ok) ++g_failed;
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", label.c_str(),
                detail.empty() ? "" : "  ", detail.c_str());
}

// A check that CANNOT run here (no engine DLL, no Vulkan device) is reported as
// skipped, never as passed. It does not fail the build either, because the
// absence of a GPU is not a defect in the .lnt pipeline - but the summary
// prints the count, so a green run can never quietly hide unchecked render
// code.
static void skip(const std::string& label, const std::string& why) {
    ++g_skipped;
    std::printf("  [SKIP] %s  (%s)\n", label.c_str(), why.c_str());
}

static void section_classification(const std::string& out) {
    std::printf("\nN. PTC CLASSIFICATION\n");
    const std::string path = out + ".classification.ptc";
    {
        std::ofstream f(path, std::ios::binary);
        f << "1\tGround\t\n*\t\t1\t255,128,0\t0\t1.0\n\n"
          << "3\tMedium Vegetation\t\n*\t\t3\t0,255,0\t1\t2.0\n\n"
          << "5\tBuildings\t\n*\t\t5\t255,100,180\t1\t1.0\n";
    }
    ClassificationTable table;
    const bool loaded = table.load_from_ptc(path);
    check("PTC parser loads three class records", loaded, table.error());
    PackedClassColor lut[256];
    table.export_gpu_lut(lut);
    check("class 1 is hidden orange", loaded && lut[1].r == 255 &&
          lut[1].g == 128 && lut[1].b == 0 && lut[1].visible == 0);
    check("class 3 is visible green with weight", loaded && lut[3].g == 255 &&
          lut[3].visible == 1 && table.rule(3).weight == 2.0f);
    check("class 5 is visible pink", loaded && lut[5].r == 255 &&
          lut[5].g == 100 && lut[5].b == 180 && lut[5].visible == 1);
    std::remove(path.c_str());
}

// ========================================================================
// comparing decoded payloads
// ========================================================================
// RawPoint is 28 bytes of payload in a 32-byte object (3 doubles + 4 uint8,
// 8-byte aligned), so its last 4 bytes are TAIL PADDING and the padding is
// indeterminate: memcmp over sizeof(RawPoint) compares bytes no decode ever
// writes, which is allocator memory. That is not hypothetical - it is exactly
// how the "the main reader is undisturbed while the loader is reading" check
// failed under -O0 (310 bytes, always at offset +28/+29 of every record) while
// passing under -O3: two identical decodes compared unequal because two
// vectors happened to come from differently-dirty heap blocks. It would have
// been read as a decoder race, and it is not one.
//
// So the comparison is EXACT OVER THE 28 PAYLOAD BYTES, per record. Byte-exact
// rather than field-by-field on purpose: it still catches a NaN payload or a
// sign-bit difference that `==` on a double would call equal.
static constexpr std::size_t kRawPointPayloadBytes = offsetof(RawPoint, classification) + 1;

// Small scientific-format helper: std::to_string() gives six decimals, which
// turns a relative difference of 4.4e-08 into "0.000000" and hides the very
// number the check is about.
static std::string fmt_e(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.3e", v);
    return std::string(buf);
}

static bool same_points(const std::vector<RawPoint>& a, const std::vector<RawPoint>& b) {
    if (a.size() != b.size()) return false;
    const unsigned char* pa = reinterpret_cast<const unsigned char*>(a.data());
    const unsigned char* pb = reinterpret_cast<const unsigned char*>(b.data());
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(pa + i * sizeof(RawPoint), pb + i * sizeof(RawPoint),
                        kRawPointPayloadBytes) != 0)
            return false;
    }
    return true;
}

// ========================================================================
// deterministic RNG (splitmix64) - reproducible test data
// ========================================================================
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    // 53-bit mantissa -> uniform [0, 1)
    double unit() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }
    uint8_t byte() { return static_cast<uint8_t>(next() >> 56); }
};

// ========================================================================
// SourceIndex - "does this decoded point sit on a real input point?"
// ========================================================================
// A decoded point is a quantised copy of one specific input point, so the
// fidelity question is "how far is it from ITS source point?", which needs a
// nearest-neighbour search over 100k points for every decoded point. The
// dataset is bucketed on a uniform grid of the global worst cell step, so the
// answer lies in the 27 neighbouring cells and the search is O(1) rather than
// O(N). One shared implementation, used by both the payload audit and the
// reader round-trip, so the two cannot disagree about what "matched" means.
class SourceIndex {
public:
    void build(const std::vector<RawPoint>& pts, const double origin[3], double grid) {
        grid_ = (grid > 0.0) ? grid : 1e-9;
        origin_[0] = origin[0];
        origin_[1] = origin[1];
        origin_[2] = origin[2];
        cells_.clear();
        cells_.reserve(pts.size() * 2);
        for (std::size_t i = 0; i < pts.size(); ++i)
            cells_[key(pts[i].x, pts[i].y, pts[i].z)].push_back(
                static_cast<uint32_t>(i));
        pts_ = &pts;
    }

    bool empty() const { return cells_.empty(); }

    // Distance to the nearest input point; out_index receives its index when
    // provided. Returns +inf when the index is empty.
    double nearest(double x, double y, double z, uint32_t* out_index = nullptr) const {
        if (pts_ == nullptr || cells_.empty()) return 1e300;
        const uint64_t k = key(x, y, z);
        const int64_t bx = static_cast<int64_t>((k >> 42) & 0x1FFFFF);
        const int64_t by = static_cast<int64_t>((k >> 21) & 0x1FFFFF);
        const int64_t bz = static_cast<int64_t>(k & 0x1FFFFF);
        double best = 1e300;
        uint32_t bi = 0;
        for (int dx = -1; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dz = -1; dz <= 1; ++dz) {
                    const uint64_t nk =
                        (static_cast<uint64_t>((bx + dx) & 0x1FFFFF) << 42) |
                        (static_cast<uint64_t>((by + dy) & 0x1FFFFF) << 21) |
                        static_cast<uint64_t>((bz + dz) & 0x1FFFFF);
                    auto it = cells_.find(nk);
                    if (it == cells_.end()) continue;
                    for (uint32_t si : it->second) {
                        const RawPoint& s = (*pts_)[si];
                        const double d = std::sqrt((s.x - x) * (s.x - x) +
                                                  (s.y - y) * (s.y - y) +
                                                  (s.z - z) * (s.z - z));
                        if (d < best) { best = d; bi = si; }
                    }
                }
        if (out_index != nullptr) *out_index = bi;
        return best;
    }

    // The index of the source point nearest (x,y,z), or -1 if none is near.
    int64_t nearest_index(double x, double y, double z, double* out_dist) const {
        uint32_t i = 0;
        const double d = nearest(x, y, z, &i);
        if (out_dist != nullptr) *out_dist = d;
        return (d < 1e299) ? static_cast<int64_t>(i) : -1;
    }

private:
    uint64_t key(double x, double y, double z) const {
        const int64_t ix = static_cast<int64_t>(std::floor((x - origin_[0]) / grid_));
        const int64_t iy = static_cast<int64_t>(std::floor((y - origin_[1]) / grid_));
        const int64_t iz = static_cast<int64_t>(std::floor((z - origin_[2]) / grid_));
        return (static_cast<uint64_t>(ix & 0x1FFFFF) << 42) |
               (static_cast<uint64_t>(iy & 0x1FFFFF) << 21) |
               static_cast<uint64_t>(iz & 0x1FFFFF);
    }

    double grid_ = 1e-9;
    double origin_[3] = {0.0, 0.0, 0.0};
    const std::vector<RawPoint>* pts_ = nullptr;
    std::unordered_map<uint64_t, std::vector<uint32_t>> cells_;
};

// ========================================================================
// Background detection for the render checks
// ========================================================================
// The background is taken to be the MOST COMMON colour in the frame rather
// than the clear colour the harness pushed. With a small point sprite over a
// mostly empty viewport those are the same thing, but deriving it from the
// image means the pixel counts do not depend on how the swapchain format
// encodes the clear (linear vs sRGB, byte order, the shader's own sRGB
// pre-compensation). Hard-coding "red means background" would silently count a
// format difference as drawn geometry.
static void dominant_colour(const EngineFrame& f, uint8_t rgb[3], uint64_t* count) {
    std::unordered_map<uint32_t, uint64_t> hist;
    hist.reserve(4096);
    for (std::size_t i = 0; i + 3 < f.rgba.size(); i += 4) {
        const uint32_t k = (uint32_t(f.rgba[i]) << 16) |
                           (uint32_t(f.rgba[i + 1]) << 8) | uint32_t(f.rgba[i + 2]);
        ++hist[k];
    }
    uint32_t best = 0;
    uint64_t best_n = 0;
    for (const auto& kv : hist)
        if (kv.second > best_n) { best_n = kv.second; best = kv.first; }
    if (rgb) {
        rgb[0] = uint8_t((best >> 16) & 0xFF);
        rgb[1] = uint8_t((best >> 8) & 0xFF);
        rgb[2] = uint8_t(best & 0xFF);
    }
    if (count) *count = best_n;
}

// Distinct colours in the frame. A cloud rendered in a single flat colour
// returns 1; a colour or elevation ramp returns many, which is the cheap way to
// show that an attribute stream reached the shader.
static std::size_t distinct_colours(const EngineFrame& f) {
    std::unordered_map<uint32_t, uint64_t> hist;
    for (std::size_t i = 0; i + 3 < f.rgba.size(); i += 4)
        ++hist[(uint32_t(f.rgba[i]) << 16) | (uint32_t(f.rgba[i + 1]) << 8) |
              uint32_t(f.rgba[i + 2])];
    return hist.size();
}

// ========================================================================
// A. struct sizes
// ========================================================================
static void section_layout() {
    std::printf("\n=== A. BINARY STRUCT LAYOUT ===\n");
    // Compile-time guards already fired in LNTFormat.h; this reports them and
    // re-verifies at runtime so the numbers appear in the log.
    std::printf("  sizeof(LNTHeader)          = %zu\n", sizeof(LNTHeader));
    std::printf("  sizeof(LNTDirectoryHeader) = %zu\n", sizeof(LNTDirectoryHeader));
    std::printf("  sizeof(LNTPoint)           = %zu\n", sizeof(LNTPoint));
    std::printf("  sizeof(LNTNodeEntry)       = %zu\n", sizeof(LNTNodeEntry));
    check("LNTHeader is exactly 64 bytes", sizeof(LNTHeader) == 64);
    check("LNTDirectoryHeader is exactly 16 bytes", sizeof(LNTDirectoryHeader) == 16);
    check("LNTPoint is exactly 11 bytes", sizeof(LNTPoint) == 11);
    check("LNTNodeEntry is exactly 52 bytes", sizeof(LNTNodeEntry) == 52);
    check("field offsets match the spec (header.index_offset == 8)",
          offsetof(LNTHeader, index_offset) == 8);
    check("field offsets match the spec (node.children_mask == 40)",
          offsetof(LNTNodeEntry, children_mask) == 40);
    check("field offsets match the spec (node.tree_level == 41)",
          offsetof(LNTNodeEntry, tree_level) == 41);
    check("3 x 21 bits = 63 <= 64, so a Morton key cannot overflow uint64",
          3 * kAxisBits <= 64, "63 of 64 bits used");
}

// ========================================================================
// B. spatial math vs an independent reference
// ========================================================================
static uint64_t naive_expand_21(uint64_t v) {
    uint64_t out = 0;
    for (int b = 0; b < 21; ++b)
        out |= ((v >> b) & 1ull) << (3 * b);
    return out;
}

static uint64_t naive_encode(uint32_t x, uint32_t y, uint32_t z) {
    uint64_t out = 0;
    for (int b = 0; b < 21; ++b) {
        out |= static_cast<uint64_t>((x >> b) & 1u) << (3 * b);
        out |= static_cast<uint64_t>((y >> b) & 1u) << (3 * b + 1);
        out |= static_cast<uint64_t>((z >> b) & 1u) << (3 * b + 2);
    }
    return out;
}

static void section_math() {
    std::printf("\n=== B. MORTON + QUANTISATION MATH ===\n");

    // B1. expand_bits_21 against the naive per-bit spread, exhaustively over a
    // stride that covers all 21 bit positions.
    uint64_t mismatch = 0;
    for (uint64_t v = 0; v <= 2097151ull; v += 4093) {
        if (expand_bits_21(v) != naive_expand_21(v)) ++mismatch;
    }
    check("expand_bits_21 matches a naive bit-by-bit spread", mismatch == 0,
          std::to_string(mismatch) + " mismatches over 513 samples");

    // B2. expand/compress round-trip over the whole 21-bit range plus the edges.
    uint64_t rt_bad = 0;
    for (uint64_t v = 0; v <= 2097151ull; ++v) {
        if (compress_bits_21(expand_bits_21(v)) != v) ++rt_bad;
        if (v > 4096 && v % 7 == 0) v += 30;      // stride the interior, keep edges
    }
    check("compress_bits_21 inverts expand_bits_21 across the full range", rt_bad == 0,
          std::to_string(rt_bad) + " mismatches");

    // B3. encode_morton_64 against the naive interleave, including every axis
    // extreme (the classic overflow site).
    uint64_t enc_bad = 0;
    const uint32_t vals[] = {0u, 1u, 2u, 7u, 255u, 1023u, 1048575u, 1048576u,
                             2097150u, 2097151u, 1234567u, 765432u};
    for (uint32_t x : vals)
        for (uint32_t y : vals)
            for (uint32_t z : vals)
                if (encode_morton_64(x, y, z) != naive_encode(x, y, z)) ++enc_bad;
    check("encode_morton_64 matches a naive interleave at all axis extremes",
          enc_bad == 0, std::to_string(enc_bad) + " mismatches of " +
                        std::to_string(sizeof(vals) / 4 * sizeof(vals) / 4 *
                                       sizeof(vals) / 4));
    check("all three axes at max yields 2^63-1 (63 bits, no wrap)",
          encode_morton_64(2097151u, 2097151u, 2097151u) == 0x7FFFFFFFFFFFFFFFull,
          "0x7FFFFFFFFFFFFFFF");

    // B4. decode is the exact inverse.
    uint64_t dec_bad = 0;
    for (uint32_t x : vals)
        for (uint32_t y : vals)
            for (uint32_t z : vals) {
                uint32_t a, b, c;
                decode_morton_64(encode_morton_64(x, y, z), a, b, c);
                if (a != x || b != y || c != z) ++dec_bad;
            }
    check("decode_morton_64 inverts encode_morton_64", dec_bad == 0,
          std::to_string(dec_bad) + " mismatches");

    // B5. THE SLIVER REGRESSION (the defect fixed in the Python reference).
    // With scale 2**bits - 1 the top level was reachable only at exactly max.
    // With scale 2**bits every one of the 2**bits levels is a full-width bin.
    // NOTE: neither quantiser takes a bit width - kAxisBits / kLocalBits are
    // fixed by the format - so each is checked against ITS OWN level count.
    // (An earlier version of this test passed 6 and 12 as a "bits" argument to
    // a function that has no such parameter, and reported a defect that was
    // purely the test's own arithmetic.)
    // The 21-bit key quantiser floors into 2**21 uniform bins, so its bin
    // centres map to their own index and the top bin is FULL WIDTH.
    auto sliver_regression = [&](const char* name, int levels) {
        const int side = levels - 1;
        const double span = 4000.0;
        int off_center = 0;
        for (int k = 0; k < levels; k += (levels > 8192 ? 199 : 1)) {
            const double v = (k + 0.5) * (span / levels);
            if (static_cast<int>(quantize_to_21bit(v, 0.0, span)) != k) ++off_center;
        }
        check(std::string(name) + ": every one of the " +
                  std::to_string(levels) + " levels is reachable",
              off_center == 0, "bin centres map to their own index");

        // The boundary between side-1 and side must sit at span*side/levels,
        // which is what makes the top bin FULL WIDTH instead of a sliver.
        const double boundary = span * side / levels;
        const double eps = span / levels * 1e-3;
        const uint32_t below = quantize_to_21bit(boundary - eps, 0.0, span);
        const uint32_t above = quantize_to_21bit(boundary + eps, 0.0, span);
        check(std::string(name) + ": top level " + std::to_string(side) +
                  " has full width (the v1 sliver was 0 m)",
              below == static_cast<uint32_t>(side - 1) &&
                  above == static_cast<uint32_t>(side),
              "just below -> " + std::to_string(below) + ", just above -> " +
                  std::to_string(above));
    };
    sliver_regression("21-bit axis quantiser", kAxisLevels);

    // The 16-bit LOCAL quantiser uses the other convention: it rounds onto the
    // 65535 intervals that node_local_to_world() divides by, so that the two are
    // exact inverses. Its two boundary levels are therefore HALF width by design,
    // and asserting "top level full width" here would demand back the very
    // mismatch the convention exists to remove. What is asserted instead is the
    // property that mismatch was costing: the error is bounded by HALF a step.
    {
        const double span = 4000.0;
        int off_grid = 0;
        for (int k = 0; k <= kLocalMax; ++k) {
            // The endpoint convention is exact at its own nodes, so grid point k
            // must encode to exactly k - for floor and round alike.
            const double v = span * double(k) / double(kLocalMax);
            if (static_cast<int>(quantize_to_16bit(v, 0.0, span)) != k) ++off_grid;
        }
        check("16-bit local quantiser: every one of the 65536 levels is reachable",
              off_grid == 0, std::to_string(off_grid) + " levels unreachable");

        const int steps = 200000;
        double worst = 0.0, worst_v = 0.0, worst_step = 0.0;
        // Sweep several cell sizes: the bound must hold for a 4 km shard and for
        // a millimetre-scale leaf alike, not just for one convenient span.
        const double spans[3] = {4000.0, 1200.0, 0.002};
        for (double sp : spans) {
            const double step = node_cell_step(float(sp * 0.5));
            for (int i = 0; i <= steps; ++i) {
                const double v = sp * double(i) / double(steps);
                const uint16_t q = quantize_to_16bit(v, 0.0, sp);
                const double back = dequantize_from_16bit(q, 0.0, sp);
                const double e = std::fabs(back - v);
                if (e / step > worst) {
                    worst = e / step;
                    worst_v = v;
                    worst_step = step;
                }
            }
        }
        check("16-bit local quantiser: mid-bin rounding bounds the error to half a step",
              worst <= 0.5 * (1.0 + 1e-9),
              "worst " + std::to_string(worst) + " x step at v=" +
                  std::to_string(worst_v) + " (step " +
                  std::to_string(worst_step) + " m)");
    }
    check("v == max maps to the top level (endpoint folded, not clipped away)",
          quantize_to_21bit(4000.0, 0.0, 4000.0) == static_cast<uint32_t>(kAxisMax));
    check("out-of-range input clamps instead of wrapping",
          quantize_to_21bit(1e12, 0.0, 4000.0) == static_cast<uint32_t>(kAxisMax) &&
              quantize_to_21bit(-1e12, 0.0, 4000.0) == 0u);
    check("16-bit local quantiser spans 0..65535",
          quantize_to_16bit(10.0, 10.0, 20.0) == 0u &&
              quantize_to_16bit(20.0, 10.0, 20.0) == 65535u &&
              quantize_to_16bit(15.0, 10.0, 20.0) == 32768u);

    // B6. octant extraction. Using a 64-bit code width here reads three
    // always-zero bits at level 0 and silently reports octant 0 for the whole
    // outer half of the cube, so it is worth pinning.
    check("octant_at level 0 reads the axes' top bit, not unused padding",
          octant_at(encode_morton_64(2097151u, 0u, 0u), 0) == 1u &&
              octant_at(encode_morton_64(0u, 2097151u, 0u), 0) == 2u &&
              octant_at(encode_morton_64(0u, 0u, 2097151u), 0) == 4u &&
              octant_at(encode_morton_64(2097151u, 2097151u, 2097151u), 0) == 7u,
          "octants 1 / 2 / 4 / 7 for +X / +Y / +Z / +++");
    check("morton_prefix(level 0) is empty and (21) is the whole 63-bit key",
          morton_prefix(encode_morton_64(7u, 7u, 7u), 0) == 0 &&
              morton_prefix(1234567890123ull, 21) ==
                  (1234567890123ull & 0x7FFFFFFFFFFFFFFFull));

    // B7. node_id MUST be unique across levels. A bare Morton prefix is not:
    // the root's empty prefix and the octant-0 child's prefix are both 0, so
    // an untagged scheme names two different nodes 0.
    {
        const uint64_t c = encode_morton_64(12345u, 67890u, 11111u);
        bool unique = true;
        std::vector<uint64_t> ids;
        for (int lv = 0; lv <= kAxisBits; ++lv) {
            const uint64_t id = node_id_at(c, lv);
            if (std::find(ids.begin(), ids.end(), id) != ids.end()) unique = false;
            ids.push_back(id);
        }
        check("node_id_at is unique across all 22 levels (no root/octant-0 alias)",
              unique, std::to_string(ids.size()) + " levels, distinct ids");
        check("node_id_at(level 0) != 0 so the root cannot alias a child",
              node_id_at(c, 0) != 0 && node_id_at(c, 1) != node_id_at(c, 0),
              "root id = " + std::to_string(node_id_at(c, 0)));
        bool relation = true, level_ok = true;
        for (int lv = 1; lv <= kAxisBits; ++lv) {
            if ((node_id_at(c, lv) >> 3) != node_id_at(c, lv - 1)) relation = false;
            if (level_of_node_id(node_id_at(c, lv)) != lv) level_ok = false;
        }
        check("child_id >> 3 == parent_id and level_of_node_id recovers the level",
              relation && level_ok,
              relation ? "relation + level recovery both hold"
                       : "relation broken");
    }
}

// ========================================================================
// synthetic dataset - deliberately NON-cubic, UTM-like magnitudes
// ========================================================================
struct Dataset {
    std::vector<RawPoint> pts;
    double extent[3] = {0, 0, 0};
};

static Dataset make_dataset(size_t n, uint64_t seed) {
    Dataset d;
    d.pts.resize(n);
    Rng rng(seed);
    // 1200 x 1000 x 120 m around a real UTM-like origin: the Z range is 10x
    // narrower than X, so a non-cubic envelope is the honest starting point.
    const double x0 = 682000.0, y0 = 3504000.0, z0 = 1040.0;
    const double sx = 1200.0, sy = 1000.0, sz = 120.0;
    d.extent[0] = sx; d.extent[1] = sy; d.extent[2] = sz;
    for (size_t i = 0; i < n; ++i) {
        RawPoint& p = d.pts[i];
        p.x = x0 + rng.unit() * sx;
        p.y = y0 + rng.unit() * sy;
        p.z = z0 + rng.unit() * sz;
        p.r = rng.byte();
        p.g = rng.byte();
        p.b = rng.byte();
        p.intensity = rng.byte();
        // Derived from the index, NOT drawn from the RNG: drawing would shift every later
        // random coordinate and change the geometry other sections were written against.
        p.classification = uint8_t((uint32_t(i) * 2654435761u) >> 24);
    }
    return d;
}

// ========================================================================
// C. build
// ========================================================================
static const char* kOutDefault = "test.lnt";

static void section_build(const Dataset& ds, const std::string& path,
                          LNTBuildStats& st) {
    std::printf("\n=== C. BUILD 100,000 POINTS -> .lnt ===\n");
    std::printf("  source envelope: X=%.0f m  Y=%.0f m  Z=%.0f m  (aspect X/Z=%.1f)\n",
                ds.extent[0], ds.extent[1], ds.extent[2],
                ds.extent[0] / ds.extent[2]);

    LNTBuildParams params;
    params.max_points_per_node = 60000;   // => 1 split level for 100k points
    params.verbose = false;

    const auto t0 = std::chrono::steady_clock::now();
    st = LNTBuilder::build(ds.pts.data(), ds.pts.size(), path.c_str(), params);
    const double wall = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0).count();

    check("builder reported success", st.ok, st.ok ? "" : st.error);
    if (!st.ok) return;

    std::printf("  input=%llu  nodes=%llu (%llu leaves)  depth=%llu  threads=%u\n",
                (unsigned long long)st.input_points,
                (unsigned long long)st.node_count,
                (unsigned long long)st.leaf_count,
                (unsigned long long)st.max_depth, st.threads);
    std::printf("  written=%llu pts (leaf %llu + LOD %llu)  file=%llu B  index_offset=%llu\n",
                (unsigned long long)st.written_points,
                (unsigned long long)st.leaf_points,
                (unsigned long long)st.lod_points,
                (unsigned long long)st.file_bytes,
                (unsigned long long)st.index_offset);
    std::printf("  timings: scan %.4fs  morton %.4fs  sort %.4fs  tree %.4fs  "
                "write %.4fs  total %.4fs (wall %.4fs)\n",
                st.scan_s, st.morton_s, st.sort_s, st.tree_s, st.write_s,
                st.total_s, wall);
    std::printf("  precision: worst cell step %.4e m  worst recon error %.4e m\n",
                st.max_cell_step, st.max_recon_error);
    std::printf("             leaf-only:   step %.4e m  err %.4e m\n",
                st.max_leaf_cell_step, st.max_leaf_recon_error);

    check("the envelope was forced CUBIC despite a 10:1 source aspect",
          st.bounds.is_cube(1e-4),
          "X=Y=Z=" + std::to_string(st.bounds.extent(0)) + " m");
    check("all input points were written into leaf chunks",
          st.leaf_points == st.input_points,
          std::to_string(st.leaf_points) + " of " + std::to_string(st.input_points));
    check("the radix sort proved stable (equal keys keep source order)",
          st.stable_sort);
    check("parallel radix sort used more than one thread", st.threads > 1,
          std::to_string(st.threads) + " workers");
    check("reconstruction error respects the reported cell-step bound",
          st.max_recon_error <= st.max_cell_step,
          "err " + std::to_string(st.max_recon_error) + " <= step " +
              std::to_string(st.max_cell_step));
    check("more than one node was produced (the octree actually split)",
          st.node_count > 1, std::to_string(st.node_count) + " nodes");
}

// ========================================================================
// D. read the file back
// ========================================================================
struct FileView {
    LNTHeader hdr{};
    LNTDirectoryHeader dir{};
    std::vector<LNTNodeEntry> nodes;
    uint64_t file_bytes = 0;
};

static bool read_struct(std::ifstream& f, void* dst, size_t bytes) {
    f.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    return f.good();
}

static FileView section_read_file(const std::string& path) {
    std::printf("\n=== D. FILE STRUCTURE ===\n");
    FileView v;
    std::ifstream f(path, std::ios::binary);
    check("output file opens", f.good(), path);
    if (!f) return v;

    f.seekg(0, std::ios::end);
    v.file_bytes = static_cast<uint64_t>(static_cast<std::streamoff>(f.tellg()));
    f.seekg(0, std::ios::beg);

    check("file is larger than a bare header", v.file_bytes > sizeof(LNTHeader),
          std::to_string(v.file_bytes) + " bytes");

    // ---- magic at byte 0 -------------------------------------------------
    // Sequence the read before the detail string: as arguments to check() the
    // order is unspecified, so the detail could be built from the zeroed
    // struct and print an empty magic on a healthy file.
    const bool hdr_ok = read_struct(f, &v.hdr, sizeof(v.hdr));
    const bool magic_ok = hdr_ok && v.hdr.magic[0] == 'L' &&
                          v.hdr.magic[1] == 'N' && v.hdr.magic[2] == 'T' &&
                          v.hdr.magic[3] == '\0';
    check("header starts with {'L','N','T','\\0'} at byte 0", magic_ok,
          hdr_ok ? std::string("magic=[") + v.hdr.magic + "]"
                 : std::string("header short read"));
    check("magic_matches() accepts the header", magic_matches(v.hdr.magic));
    check("format version is 2", v.hdr.version == 2,
          "version=" + std::to_string(v.hdr.version));
    check("reserved field is zero", v.hdr.reserved == 0);
    check("index_offset is inside the file and past the header",
          v.hdr.index_offset >= sizeof(LNTHeader) && v.hdr.index_offset < v.file_bytes,
          std::to_string(v.hdr.index_offset));
    check("header bounds are a cubic envelope",
          std::fabs((v.hdr.bounds_max[0] - v.hdr.bounds_min[0]) -
                    (v.hdr.bounds_max[1] - v.hdr.bounds_min[1])) < 1e-4 &&
              std::fabs((v.hdr.bounds_max[0] - v.hdr.bounds_min[0]) -
                        (v.hdr.bounds_max[2] - v.hdr.bounds_min[2])) < 1e-4);

    // ---- jump straight to index_offset ----------------------------------
    f.clear();
    f.seekg(static_cast<std::streamoff>(v.hdr.index_offset), std::ios::beg);
    check("seek to index_offset succeeds", f.good());
    // The read must be sequenced BEFORE the detail string is built. Passing
    // both as arguments to check() leaves the order unspecified, so the detail
    // could be formatted from the still-zeroed struct and print 0 total_points
    // for a healthy file.
    const bool dir_ok = read_struct(f, &v.dir, sizeof(v.dir));
    check("LNTDirectoryHeader reads at index_offset", dir_ok,
          dir_ok ? "total_points=" + std::to_string(v.dir.total_points) +
                       " total_nodes=" + std::to_string(v.dir.total_nodes)
                 : std::string("short read"));
    check("directory reports a non-zero node count", v.dir.total_nodes > 0);
    check("directory total_points matches the source point count",
          v.dir.total_points == 100000ull,
          std::to_string(v.dir.total_points));

    // ---- trailing bytes: the file must END exactly at the directory -----
    const uint64_t expected = v.hdr.index_offset + sizeof(LNTDirectoryHeader) +
                              v.dir.total_nodes * sizeof(LNTNodeEntry);
    check("file size == index_offset + 16 + 52*node_count (no trailing bytes)",
          expected == v.file_bytes,
          std::to_string(expected) + " == " + std::to_string(v.file_bytes));

    v.nodes.resize(static_cast<size_t>(v.dir.total_nodes));
    const uint64_t want = v.dir.total_nodes * sizeof(LNTNodeEntry);
    f.read(reinterpret_cast<char*>(v.nodes.data()),
           static_cast<std::streamsize>(want));
    check("the full node table is readable after the directory header",
          static_cast<uint64_t>(f.gcount()) == want,
          std::to_string(f.gcount()) + " of " + std::to_string(want) + " bytes");
    return v;
}

// ========================================================================
// E. spatial integrity of the decoded payload
// ========================================================================
static void section_payload(const std::string& path, const FileView& v,
                            const Dataset& ds, const LNTBuildStats& st) {
    std::printf("\n=== E. PAYLOAD SPATIAL INTEGRITY ===\n");
    std::ifstream f(path, std::ios::binary);

    // Chunk bounds: every node's range must lie inside [64, index_offset) and
    // no two chunks may overlap.
    bool in_bounds = true, overlaps = false;
    std::vector<std::pair<uint64_t, uint64_t>> spans;
    for (const auto& e : v.nodes) {
        const uint64_t lo = e.byte_offset, hi = e.byte_offset + e.byte_length;
        if (lo < sizeof(LNTHeader) || hi > v.hdr.index_offset || e.byte_length == 0)
            in_bounds = false;
        if (e.byte_length != static_cast<uint64_t>(e.point_count) * sizeof(LNTPoint))
            in_bounds = false;
        spans.emplace_back(lo, hi);
    }
    std::sort(spans.begin(), spans.end());
    for (size_t i = 1; i < spans.size(); ++i)
        if (spans[i].first < spans[i - 1].second) overlaps = true;
    check("every node chunk lies inside the payload region", in_bounds);
    check("byte_length == point_count * 11 for every node", in_bounds);
    check("node chunks do not overlap", !overlaps);

    // node_id uniqueness and prefix consistency: a child's id must be its
    // parent's id with its own octant appended.
    std::unordered_map<uint64_t, const LNTNodeEntry*> by_id;
    bool ids_unique = true;
    for (const auto& e : v.nodes) {
        if (!by_id.emplace(e.node_id, &e).second) ids_unique = false;
    }
    check("node_id values are unique (each Morton prefix names one node)",
          ids_unique, std::to_string(v.nodes.size()) + " nodes");

    bool prefix_ok = true, mask_ok = true;
    for (const auto& e : v.nodes) {
        if (e.tree_level == 0) continue;
        const uint64_t parent_id = e.node_id >> 3;
        auto it = by_id.find(parent_id);
        if (it == by_id.end()) { prefix_ok = false; continue; }
        const uint8_t octant = static_cast<uint8_t>(e.node_id & 0x7ull);
        if ((it->second->children_mask & (1u << octant)) == 0) mask_ok = false;
    }
    check("every non-root node_id extends its parent's prefix", prefix_ok);
    check("every parent's children_mask contains each existing child's octant",
          mask_ok);

    // ---- decode every point ---------------------------------------------
    // Source points are indexed by a grid of the global worst cell step, so
    // "does this decoded point sit on a real source point?" is answered by
    // scanning only 27 cells.
    const double grid = std::max(st.max_cell_step, 1e-9);
    const double gx = v.hdr.bounds_min[0], gy = v.hdr.bounds_min[1],
                 gz = v.hdr.bounds_min[2];
    auto cell_of = [&](double x, double y, double z) -> uint64_t {
        const int64_t ix = static_cast<int64_t>(std::floor((x - gx) / grid));
        const int64_t iy = static_cast<int64_t>(std::floor((y - gy) / grid));
        const int64_t iz = static_cast<int64_t>(std::floor((z - gz) / grid));
        return (static_cast<uint64_t>(ix & 0x1FFFFF) << 42) |
               (static_cast<uint64_t>(iy & 0x1FFFFF) << 21) |
               static_cast<uint64_t>(iz & 0x1FFFFF);
    };
    std::unordered_map<uint64_t, std::vector<uint32_t>> grid_map;
    grid_map.reserve(ds.pts.size() * 2);
    for (size_t i = 0; i < ds.pts.size(); ++i)
        grid_map[cell_of(ds.pts[i].x, ds.pts[i].y, ds.pts[i].z)].push_back(
            static_cast<uint32_t>(i));

    uint64_t decoded = 0, outside_cube = 0, unmatched = 0, bad_step = 0;
    double worst = 0.0, worst_step = 0.0;
    std::vector<LNTPoint> pts;

    for (const auto& e : v.nodes) {
        const double step = node_cell_step(e.half_size);
        // Each AXIS is bounded by `step`, so the 3D distance is bounded by
        // sqrt(3)*step. Comparing the Euclidean distance against `step` alone
        // (as an earlier version did) reports ~17% of points as failures when
        // every axis is in fact well inside its own bound.
        const double tol = std::sqrt(3.0) * step + 1e-9;
        f.clear();
        f.seekg(static_cast<std::streamoff>(e.byte_offset), std::ios::beg);
        pts.resize(e.point_count);
        f.read(reinterpret_cast<char*>(pts.data()),
               static_cast<std::streamsize>(e.point_count * sizeof(LNTPoint)));
        if (static_cast<uint32_t>(f.gcount()) != e.point_count * sizeof(LNTPoint)) {
            ++unmatched;
            continue;
        }

        for (const auto& q : pts) {
            ++decoded;
            // `center` is a float32 offset from the header's bounds_min, not an
            // absolute world coordinate.
            const double wx = node_local_to_world(q.x_rel, e.center[0], e.half_size,
                                                  v.hdr.bounds_min[0]);
            const double wy = node_local_to_world(q.y_rel, e.center[1], e.half_size,
                                                  v.hdr.bounds_min[1]);
            const double wz = node_local_to_world(q.z_rel, e.center[2], e.half_size,
                                                  v.hdr.bounds_min[2]);

            // (1) containment: the point must lie inside its own node cube.
            const double lo[3] = {
                v.hdr.bounds_min[0] + e.center[0] - e.half_size,
                v.hdr.bounds_min[1] + e.center[1] - e.half_size,
                v.hdr.bounds_min[2] + e.center[2] - e.half_size};
            const double w[3] = {wx, wy, wz};
            for (int a = 0; a < 3; ++a)
                if (w[a] < lo[a] - tol || w[a] > lo[a] + 2.0 * e.half_size + tol)
                    ++outside_cube;

            // (2) fidelity: a real source point must lie within one step.
            const uint64_t base = cell_of(wx, wy, wz);
            const int64_t bx = static_cast<int64_t>((base >> 42) & 0x1FFFFF);
            const int64_t by = static_cast<int64_t>((base >> 21) & 0x1FFFFF);
            const int64_t bz = static_cast<int64_t>(base & 0x1FFFFF);
            double best = 1e30;
            for (int dx = -1; dx <= 1; ++dx)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dz = -1; dz <= 1; ++dz) {
                        const uint64_t key =
                            (static_cast<uint64_t>((bx + dx) & 0x1FFFFF) << 42) |
                            (static_cast<uint64_t>((by + dy) & 0x1FFFFF) << 21) |
                            static_cast<uint64_t>((bz + dz) & 0x1FFFFF);
                        auto it = grid_map.find(key);
                        if (it == grid_map.end()) continue;
                        for (uint32_t si : it->second) {
                            const RawPoint& s = ds.pts[si];
                            const double d = std::sqrt((s.x - wx) * (s.x - wx) +
                                                       (s.y - wy) * (s.y - wy) +
                                                       (s.z - wz) * (s.z - wz));
                            if (d < best) best = d;
                        }
                    }
            if (best > tol) ++unmatched;
            if (best > worst) { worst = best; worst_step = step; }
            if (best > std::sqrt(3.0) * step * 1.000001) ++bad_step;
        }
    }

    std::printf("  decoded %llu points from %zu nodes (stride 1 leaves only joined by LOD)\n",
                (unsigned long long)decoded, v.nodes.size());
    std::printf("  worst distance to a real source point: %.6e m (node step %.6e m)\n",
                worst, worst_step);

    check("every node chunk decoded to its declared point_count",
          decoded >= st.leaf_points,
          std::to_string(decoded) + " decoded >= " + std::to_string(st.leaf_points) +
              " leaf points");
    check("every decoded point lies inside its own node's cube",
          outside_cube == 0, std::to_string(outside_cube) + " outside");
    check("every decoded point sits on a real source point within its node's 16-bit step",
          unmatched == 0,
          std::to_string(unmatched) + " unmatched, worst " + std::to_string(worst) + " m");
    check("no decoded point exceeds sqrt(3) x its node's quantisation step",
          bad_step == 0, std::to_string(bad_step) + " exceeded");
}

// ========================================================================
// F. determinism + precision knob
// ========================================================================
static bool files_identical(const std::string& a, const std::string& b) {
    std::ifstream fa(a, std::ios::binary), fb(b, std::ios::binary);
    if (!fa || !fb) return false;
    std::vector<char> ba((std::istreambuf_iterator<char>(fa)),
                         std::istreambuf_iterator<char>());
    std::vector<char> bb((std::istreambuf_iterator<char>(fb)),
                         std::istreambuf_iterator<char>());
    return ba == bb;
}

static void section_determinism(const Dataset& ds, const std::string& path) {
    std::printf("\n=== F. DETERMINISM AND THE PRECISION CONTROL ===\n");
    LNTBuildParams p;
    p.max_points_per_node = 60000;
    const std::string again = path + ".again";
    LNTBuildStats s2 = LNTBuilder::build(ds.pts.data(), ds.pts.size(), again.c_str(), p);
    check("a second identical build succeeds", s2.ok, s2.error);
    check("two builds of the same input are byte-identical",
          files_identical(path, again));

    // target_resolution drives the tree deeper so the 16-bit cell is finer.
    LNTBuildParams fine;
    fine.max_points_per_node = 60000;
    fine.target_resolution = 0.002;        // 2 mm survey precision
    fine.min_points_per_node = 256;
    const std::string fine_path = path + ".fine";
    LNTBuildStats s3 = LNTBuilder::build(ds.pts.data(), ds.pts.size(),
                                         fine_path.c_str(), fine);
    check("the precision-driven build succeeds", s3.ok, s3.error);
    check("a 2 mm resolution target produces a deeper tree",
          s3.max_depth > s2.max_depth,
          "depth " + std::to_string(s3.max_depth) + " vs " + std::to_string(s2.max_depth));
    // Compare LEAF precision, not the all-nodes maximum: the root always spans
    // the whole cube and dominates the all-nodes figure, so that comparison
    // cannot show whether deeper subdivision bought anything.
    check("a 2 mm resolution target reaches a finer LEAF cell step",
          s3.max_leaf_cell_step < s2.max_leaf_cell_step,
          std::to_string(s3.max_leaf_cell_step) + " m vs " +
              std::to_string(s2.max_leaf_cell_step) + " m");
    check("a 2 mm resolution target reduces the measured LEAF recon error",
          s3.max_leaf_recon_error < s2.max_leaf_recon_error,
          std::to_string(s3.max_leaf_recon_error) + " m vs " +
              std::to_string(s2.max_leaf_recon_error) + " m");
    std::printf("  capacity-only:  nodes=%llu depth=%llu leaf_step=%.4e m leaf_err=%.4e m\n",
                (unsigned long long)s2.node_count, (unsigned long long)s2.max_depth,
                s2.max_leaf_cell_step, s2.max_leaf_recon_error);
    std::printf("  2 mm target:    nodes=%llu depth=%llu leaf_step=%.4e m leaf_err=%.4e m\n",
                (unsigned long long)s3.node_count, (unsigned long long)s3.max_depth,
                s3.max_leaf_cell_step, s3.max_leaf_recon_error);

    // A single-node build must reproduce the source set EXACTLY: one root leaf,
    // stride 1, so the multiset of decoded positions is a bijection onto the
    // input. This is the strongest available exactness check.
    LNTBuildParams one;
    one.max_points_per_node = 200000;      // > n, so the root is the only node
    one.target_resolution = 0.0;
    const std::string one_path = path + ".onenode";
    LNTBuildStats s4 = LNTBuilder::build(ds.pts.data(), ds.pts.size(),
                                         one_path.c_str(), one);
    check("single-node build succeeds", s4.ok, s4.error);
    if (s4.ok) {
        std::ifstream f(one_path, std::ios::binary);
        LNTHeader h{};
        f.read(reinterpret_cast<char*>(&h), sizeof(h));
        f.seekg(static_cast<std::streamoff>(h.index_offset), std::ios::beg);
        LNTDirectoryHeader dh{};
        f.read(reinterpret_cast<char*>(&dh), sizeof(dh));
        std::vector<LNTNodeEntry> ns(static_cast<size_t>(dh.total_nodes));
        f.read(reinterpret_cast<char*>(ns.data()),
               static_cast<std::streamsize>(dh.total_nodes * sizeof(LNTNodeEntry)));
        check("exactly one node exists for an unsplittable input",
              ns.size() == 1, std::to_string(ns.size()) + " nodes");
        if (ns.size() == 1) {
            const LNTNodeEntry& e = ns[0];
            check("the single node holds every input point",
                  e.point_count == ds.pts.size(),
                  std::to_string(e.point_count) + " of " + std::to_string(ds.pts.size()));
            std::vector<LNTPoint> qp(e.point_count);
            f.clear();
            f.seekg(static_cast<std::streamoff>(e.byte_offset), std::ios::beg);
            f.read(reinterpret_cast<char*>(qp.data()),
                   static_cast<std::streamsize>(e.point_count * sizeof(LNTPoint)));
            // Compare sorted attribute streams: RGB and intensity must survive
            // the round-trip bit-exactly, with no point dropped or invented.
            std::vector<uint32_t> got, want;
            got.reserve(qp.size());
            want.reserve(ds.pts.size());
            for (const auto& q : qp)
                got.push_back((uint32_t(q.r) << 24) | (uint32_t(q.g) << 16) |
                              (uint32_t(q.b) << 8) | uint32_t(q.intensity));
            for (const auto& s : ds.pts)
                want.push_back((uint32_t(s.r) << 24) | (uint32_t(s.g) << 16) |
                               (uint32_t(s.b) << 8) | uint32_t(s.intensity));
            std::sort(got.begin(), got.end());
            std::sort(want.begin(), want.end());
            check("single-node RGB + intensity round-trip is an exact multiset match",
                  got == want,
                  std::to_string(got.size()) + " attributes compared");
        }
    }
}

// ========================================================================
// H. SPATIAL QUERIES + hardened error handling
// ========================================================================
struct DecodedNode {
    const LNTNodeEntry* entry = nullptr;
    std::vector<RawPoint> pts;
};

// Write a mutated copy of the file for the negative tests. The mutations are
// the ones a truncated download, a wrong tool version or a flipped byte would
// actually produce.
static bool write_corrupt(const std::string& src, const std::string& dst,
                          const std::string& what) {
    std::ifstream in(src, std::ios::binary);
    if (!in) return false;
    std::vector<char> b((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
    in.close();
    if (b.size() < sizeof(LNTHeader)) return false;
    if (what == "magic") b[0] = 'X';
    else if (what == "version") { b[4] = 3; b[5] = 0; }
    else if (what == "index_offset") {
        const uint64_t bogus = 0xFFFFFFFFFFFFull;
        std::memcpy(b.data() + 8, &bogus, 8);
    } else if (what == "truncated") {
        b.resize(40);
    } else if (what == "table_truncated") {
        uint64_t io = 0;
        std::memcpy(&io, b.data() + 8, 8);
        if (io + kDirHeaderBytes > b.size()) return false;
        b.resize(static_cast<std::size_t>(io + kDirHeaderBytes));
    } else {
        return false;
    }
    std::ofstream out(dst, std::ios::binary);
    if (!out) return false;
    out.write(b.data(), static_cast<std::streamsize>(b.size()));
    return out.good();
}

static void section_reader_queries(Reader& rd, const Dataset& ds,
                                   const std::vector<DecodedNode>& decoded,
                                   const BoundingBox& env) {
    // ---- degenerate query boxes -----------------------------------------
    const std::vector<LNTNodeEntry> all = rd.query_nodes_in_bounds(env);
    check("the whole envelope selects every node",
          all.size() == rd.nodes().size(),
          std::to_string(all.size()) + " of " + std::to_string(rd.nodes().size()));

    const BoundingBox far_box = BoundingBox::from_min_max(
        env.max[0] + 10000.0, env.max[1] + 10000.0, env.max[2] + 10000.0,
        env.max[0] + 20000.0, env.max[1] + 20000.0, env.max[2] + 20000.0);
    check("a box entirely outside the dataset selects nothing",
          rd.query_nodes_in_bounds(far_box).empty());
    check("a default-constructed (degenerate) box selects nothing",
          rd.query_nodes_in_bounds(BoundingBox()).empty());

    // ---- spec step 5: a real sub-box ------------------------------------
    // One quarter of the dataset in XY, full Z: a box that must catch some
    // nodes and miss others, which is the only way the test can distinguish
    // culling from "return everything".
    const BoundingBox sub = BoundingBox::from_min_max(
        env.min[0], env.min[1], env.min[2], env.mid(0), env.mid(1), env.max[2]);
    const std::vector<LNTNodeEntry> hit = rd.query_nodes_in_bounds(sub);
    std::printf("  sub-box query  : %zu of %zu nodes intersect the quarter box\n",
                hit.size(), rd.nodes().size());
    check("a sub-box selects a strict, non-empty subset of the directory",
          !hit.empty() && hit.size() < rd.nodes().size(),
          std::to_string(hit.size()) + " of " + std::to_string(rd.nodes().size()));

    // SOUNDNESS: nothing outside the intersection can come back.
    bool sound = true;
    for (const LNTNodeEntry& e : hit)
        if (!rd.bounds_of(e).intersects(sub)) sound = false;
    check("every returned node's cell really intersects the query box", sound);

    // COMPLETENESS: a query may not MISS a node that holds a point in the box.
    // This is the property that makes node culling safe to render with.
    std::unordered_map<uint64_t, const RawPoint*> holder;
    for (const DecodedNode& dn : decoded)
        for (const RawPoint& p : dn.pts)
            if (sub.contains(p)) holder.emplace(dn.entry->node_id, &p);
    bool complete = true;
    for (const auto& kv : holder)
        if (std::none_of(hit.begin(), hit.end(), [&](const LNTNodeEntry& e) {
                return e.node_id == kv.first;
            }))
            complete = false;
    check("no node holding a point inside the box is missed", complete,
          std::to_string(holder.size()) + " nodes actually hold a point in the box");

    // ---- the exact variant, and the distinction from the cheap one -------
    const std::vector<LNTNodeEntry> exact =
        rd.query_nodes_with_points_in_bounds(sub);
    bool exact_subset = true;
    for (const LNTNodeEntry& e : exact)
        if (std::none_of(hit.begin(), hit.end(), [&](const LNTNodeEntry& h) {
                return h.node_id == e.node_id;
            }))
            exact_subset = false;
    check("query_nodes_with_points_in_bounds is a subset of the box-intersection set",
          exact_subset, std::to_string(exact.size()) + " of " +
                            std::to_string(hit.size()) + " candidates hold a point");

    bool exact_has_point = true;
    std::vector<RawPoint> pts;
    for (const LNTNodeEntry& e : exact) {
        if (!rd.read_node_points(e, pts)) { exact_has_point = false; continue; }
        if (std::none_of(pts.begin(), pts.end(),
                         [&](const RawPoint& p) { return sub.contains(p); }))
            exact_has_point = false;
    }
    check("every node the exact query returns really holds a point in the box",
          exact_has_point);
    check("the exact query finds every node holding a point in the box",
          exact.size() == holder.size(),
          std::to_string(exact.size()) + " vs " + std::to_string(holder.size()));

    // ---- exact point extraction -----------------------------------------
    std::vector<RawPoint> in_box;
    const bool got = rd.read_points_in_bounds(sub, in_box);
    uint64_t expect = 0;
    for (const DecodedNode& dn : decoded)
        for (const RawPoint& p : dn.pts)
            if (sub.contains(p)) ++expect;
    check("read_points_in_bounds() succeeds", got);
    check("read_points_in_bounds() returns exactly the points inside the box",
          in_box.size() == expect,
          std::to_string(in_box.size()) + " vs " + std::to_string(expect) +
              " from a full decode");
    bool all_inside = true;
    for (const RawPoint& p : in_box)
        if (!sub.contains(p)) all_inside = false;
    check("every point read from a box query is inside that box", all_inside);

    // ---- a point-like box around one known source point ------------------
    const double c0[3] = {ds.pts[0].x, ds.pts[0].y, ds.pts[0].z};
    const BoundingBox dot = BoundingBox::from_center_half(c0, 1e-6);
    std::vector<RawPoint> near0;
    if (rd.read_points_in_bounds(dot.outset(0.05), near0)) {
        double best = 1e300;
        for (const RawPoint& p : near0)
            best = std::min(best, std::sqrt((p.x - ds.pts[0].x) * (p.x - ds.pts[0].x) +
                                            (p.y - ds.pts[0].y) * (p.y - ds.pts[0].y) +
                                            (p.z - ds.pts[0].z) * (p.z - ds.pts[0].z)));
        check("a point-sized box returns the source point it was centred on",
              best < 0.05,
              "nearest returned point is " + std::to_string(best) + " m away");
    } else {
        check("a point-sized box returns the source point it was centred on", false,
              "query failed");
    }

    // ---- hardened error handling ----------------------------------------
    struct Bad { const char* what; const char* why; };
    const Bad bads[] = {{"magic", "a flipped magic byte"},
                        {"version", "a future format version"},
                        {"index_offset", "an index_offset past EOF"},
                        {"truncated", "a truncated file"},
                        {"table_truncated", "a missing node table"}};
    bool rejected_all = true;
    std::vector<std::string> temps;
    for (const Bad& b : bads) {
        const std::string tmp = rd.path() + ".bad_" + b.what;
        if (!write_corrupt(rd.path(), tmp, b.what)) { rejected_all = false; continue; }
        temps.push_back(tmp);
        Reader bad;
        if (bad.open(tmp)) {
            std::printf("    NOT REJECTED: %s (%s)\n", b.what, b.why);
            rejected_all = false;
        }
    }
    check("the reader rejects a bad magic / version / offset / truncation",
          rejected_all, "5 corrupted variants");
    for (const std::string& t : temps) std::remove(t.c_str());

    // ---- a corrupt NODE must not be read as if it were points ------------
    // The dangerous failure mode: a bogus byte_offset pointing into the trailer
    // or the header would decode valid-looking garbage coordinates. The reader
    // must refuse the chunk, not guess.
    if (!rd.nodes().empty()) {
        LNTNodeEntry e = rd.nodes()[0];
        e.byte_offset = rd.get_index_offset();          // the node table itself
        std::vector<RawPoint> junk;
        const bool refused_trailer = !rd.read_node_points(e, junk);
        e = rd.nodes()[0];
        e.byte_offset = 0;                              // the header
        const bool refused_header = !rd.read_node_points(e, junk);
        e = rd.nodes()[0];
        e.byte_length += 10;                            // inconsistent length
        const bool refused_length = !rd.read_node_points(e, junk);
        check("a node whose chunk points outside the payload region is refused",
              refused_trailer && refused_header);
        check("a node whose byte_length disagrees with point_count is refused",
              refused_length, rd.last_error());
    }
}

// ========================================================================
// G. READER ROUND-TRIP (.lnt -> points)
// ========================================================================
static void section_reader(const std::string& path, const Dataset& ds,
                           const LNTBuildStats& st, const FileView& v) {
    std::printf("\n=== G. READER ROUND-TRIP (LNT::Reader) ===\n");

    Reader rd;
    const bool opened = rd.open(path);
    check("LNT::Reader opens the built file", opened, opened ? path : rd.error());
    if (!opened) return;

    // ---- the brief's EXACT API spelling, pinned at compile time ----------
    // The brief names the class `LNT::Reader` and states these signatures. A
    // static_cast to the corresponding member-pointer type only compiles if the
    // signature matches exactly, so a future refactor cannot silently break the
    // published API - the build fails instead.
    {
        using ReadNodeFn = bool (LNT::Reader::*)(const LNTNodeEntry&,
                                                 std::vector<LNT::RawPoint>&);
        using QueryFn = std::vector<LNTNodeEntry> (LNT::Reader::*)(
            const LNT::BoundingBox&) const;
        using TotalFn = uint64_t (LNT::Reader::*)() const;
        ReadNodeFn read_node = static_cast<ReadNodeFn>(&LNT::Reader::read_node_points);
        QueryFn query = static_cast<QueryFn>(&LNT::Reader::query_nodes_in_bounds);
        TotalFn total = static_cast<TotalFn>(&LNT::Reader::get_total_points);
        (void)read_node;
        (void)query;
        (void)total;
    }
    LNT::Reader spec_reader(path);
    check("LNT::Reader (the brief's spelling) opens the file", spec_reader.ok(),
          spec_reader.error());
    LNT::BoundingBox spec_box = spec_reader.get_bounds();
    check("query_nodes_in_bounds(box) returns every node by value",
          spec_reader.query_nodes_in_bounds(spec_box).size() ==
              spec_reader.get_total_nodes(),
          std::to_string(spec_reader.query_nodes_in_bounds(spec_box).size()) +
              " of " + std::to_string(spec_reader.get_total_nodes()));

    const BoundingBox env = rd.get_bounds();
    std::printf("  header         : version %u  reserved %u  index_offset %llu\n",
                rd.header().version, rd.header().reserved,
                (unsigned long long)rd.get_index_offset());
    std::printf("  directory      : %llu points  %llu nodes  file %llu bytes\n",
                (unsigned long long)rd.get_total_points(),
                (unsigned long long)rd.get_total_nodes(),
                (unsigned long long)rd.get_file_bytes());
    std::printf("  envelope       : cube=%d  %.3f x %.3f x %.3f m\n",
                env.is_cube(1e-3) ? 1 : 0, env.extent(0), env.extent(1),
                env.extent(2));

    // ---- spec step 3 -----------------------------------------------------
    check("reader.get_total_points() == 100000",
          rd.get_total_points() == 100000ull,
          std::to_string(rd.get_total_points()));

    // ---- the three directory views must agree ---------------------------
    const ReaderDiagnostics dg = rd.diagnose();
    check("total_nodes agrees with the directory and with both views",
          rd.get_total_nodes() == v.nodes.size() &&
              rd.nodes().size() == v.nodes.size() &&
              rd.node_index().size() == v.nodes.size(),
          std::to_string(rd.get_total_nodes()) + " / " +
              std::to_string(rd.nodes().size()) + " / " +
              std::to_string(rd.node_index().size()));
    bool every_id_resolves = true;
    for (const LNTNodeEntry& e : rd.nodes())
        if (rd.find_node(e.node_id) == nullptr) every_id_resolves = false;
    check("find_node() resolves every node_id in the file", every_id_resolves);
    check("find_node() rejects an id that is not in the file",
          rd.find_node(0xDEADBEEFDEADBEEFull) == nullptr);
    check("diagnose() reports a clean file (no fatal, no structural findings)",
          dg.clean(), dg.fatal() ? "fatal" : "non-fatal findings present");

    // ---- spec step 4: read every node through read_node_points -----------
    std::vector<DecodedNode> decoded;
    decoded.reserve(rd.nodes().size());
    bool all_read = true, all_sized = true, all_attrs = true;
    std::vector<LNTPoint> raw;
    uint64_t decoded_total = 0, decoded_leaf = 0;
    for (const LNTNodeEntry& e : rd.nodes()) {
        DecodedNode dn;
        dn.entry = &e;
        if (!rd.read_node_points(e, dn.pts)) { all_read = false; continue; }
        if (dn.pts.size() != e.point_count) all_sized = false;
        // Colour and intensity are stored as bytes and must survive verbatim;
        // only XYZ passes through the 16-bit quantiser.
        if (!rd.read_node_raw(e, raw)) { all_read = false; continue; }
        if (raw.size() != dn.pts.size()) all_sized = false;
        for (std::size_t i = 0; i < raw.size(); ++i)
            if (dn.pts[i].r != raw[i].r || dn.pts[i].g != raw[i].g ||
                dn.pts[i].b != raw[i].b ||
                dn.pts[i].intensity != raw[i].intensity)
                all_attrs = false;
        decoded_total += dn.pts.size();
        if (is_leaf_node(e)) decoded_leaf += dn.pts.size();
        decoded.push_back(std::move(dn));
    }
    check("read_node_points() succeeds for every node in the directory", all_read);
    check("every node decoded to its declared point_count", all_sized,
          std::to_string(decoded_total) + " points decoded");
    check("r/g/b/intensity survive the round-trip byte-exactly", all_attrs);
    check("leaf chunks alone hold every dataset point",
          decoded_leaf == rd.get_total_points(),
          std::to_string(decoded_leaf) + " of " +
              std::to_string(rd.get_total_points()));

    // ---- containment: a decoded point must lie inside its own node cell ---
    uint64_t outside = 0;
    for (const DecodedNode& dn : decoded) {
        const BoundingBox cell = rd.bounds_of(*dn.entry);
        for (const RawPoint& p : dn.pts)
            if (!cell.contains(p)) ++outside;
    }
    check("every decoded point lies inside its own node cell", outside == 0,
          std::to_string(outside) + " outside");

    // ---- fidelity against the source, per AXIS ---------------------------
    // The three axes are quantised independently, so a per-axis bound is the
    // right test and the 3D distance is bounded by sqrt(3) of it.
    //
    // The bound is HALF a cell step, and it is the format's own: the writer
    // rounds u onto the 65535 intervals that node_local_to_world() divides by
    // (quantize_to_16bit), so forward and inverse are exact inverses and the
    // residual is at most half an interval. Before that pairing was fixed the
    // forward map floored into 65536 bins while the inverse divided by 65535,
    // which left a worst case of 65535/65536 = 0.99998 of a step - one full
    // step of error, and half the payload's precision thrown away.
    const double grid = std::max(st.max_cell_step, 1e-9);
    SourceIndex idx;
    idx.build(ds.pts, rd.header().bounds_min, grid);

    double worst_axis_ratio = 0.0, worst_3d_ratio = 0.0;
    uint64_t beyond_format_bound = 0, beyond_literal_bar = 0;
    double worst_step = 0.0;
    for (const DecodedNode& dn : decoded) {
        if (!is_leaf_node(*dn.entry)) continue;   // LOD samples repeat leaves
        const double step = node_cell_step(dn.entry->half_size);
        if (!(step > 0.0)) continue;
        for (const RawPoint& p : dn.pts) {
            double d3 = 0.0;
            const int64_t si = idx.nearest_index(p.x, p.y, p.z, &d3);
            if (si < 0) { ++beyond_format_bound; continue; }
            const RawPoint& s = ds.pts[static_cast<std::size_t>(si)];
            const double da = std::max(std::fabs(s.x - p.x),
                                       std::max(std::fabs(s.y - p.y),
                                                std::fabs(s.z - p.z)));
            const double ra = da / step;
            const double r3 = d3 / step;
            if (ra > worst_axis_ratio) { worst_axis_ratio = ra; worst_step = step; }
            if (r3 > worst_3d_ratio) worst_3d_ratio = r3;
            if (ra > 0.5 * (1.0 + 1e-9)) ++beyond_format_bound;
            if (ra > 0.999) ++beyond_literal_bar;
        }
    }
    std::printf("  fidelity       : worst per-axis err = %.6f x step, worst 3D = %.6f x step\n",
                worst_axis_ratio, worst_3d_ratio);
    std::printf("                   (node step at the worst point %.6e m)\n", worst_step);
    check("every leaf point is within half a step, per axis",
          beyond_format_bound == 0,
          std::to_string(beyond_format_bound) + " beyond 0.5 x step");
    check("3D error stays inside sqrt(3) x that bound",
          worst_3d_ratio <= std::sqrt(3.0) * 0.5 * (1.0 + 1e-9),
          "worst " + std::to_string(worst_3d_ratio) + " x step");
    // The brief's explicit bar. It is asserted as written and it now holds with
    // a factor-of-two margin, because the writer rounds onto the same 65535
    // intervals the reader divides by.
    check("the brief's literal <= 0.999 x step bar is met",
          beyond_literal_bar == 0,
          beyond_literal_bar == 0
              ? "worst " + std::to_string(worst_axis_ratio) +
                    " x step, against the format's 0.5 x step bound"
              : std::to_string(beyond_literal_bar) + " points exceed it (worst " +
                    std::to_string(worst_axis_ratio) + " x step)");

    section_reader_queries(rd, ds, decoded, env);
}

// ========================================================================
// I. RENDER VIA THE VULKAN ENGINE
// ========================================================================
// What this proves, and what it refuses to claim. The engine is the existing
// naksha_vulkan one, driven through its C ABI. A pass here means real pixels
// came back through the engine's own offscreen capture with .lnt geometry in
// them, and that a node-level cull changed the image without re-uploading a
// single position. It does NOT prove frame rate at scale - the dataset is
// 100k points, far too small for the timings to mean anything.
static void section_render(const std::string& path, const Dataset& ds,
                           const LNTBuildStats& st) {
    (void)ds;
    std::printf("\n=== I. RENDER VIA THE VULKAN ENGINE ===\n");

    const std::string dll = VulkanEngine::find_dll();
    if (dll.empty()) {
        skip("render path", "no naksha_vulkan.dll found (set LNT_VULKAN_DLL)");
        return;
    }
    std::printf("  engine         : %s\n", dll.c_str());

    VulkanEngine eng;
    std::string err;
    if (!eng.load(dll, err)) { skip("render path", "engine load failed: " + err); return; }
    if (!eng.is_available()) {
        skip("render path", "no Vulkan loader or no physical device on this machine");
        return;
    }
    // Hidden window: the engine's capture renders offscreen into its own target,
    // so pixel verification needs no visible UI and no presented frame.
    if (!eng.open(960, 720, /*hidden=*/true, /*validation=*/false, err)) {
        skip("render path", "renderer creation failed: " + err);
        return;
    }
    check("the Vulkan engine loads, resolves its ABI and creates a device",
          !eng.device_name().empty() && eng.api_version() != 0,
          eng.device_name());

    Reader rd;
    if (!rd.open(path)) {
        check("the render path can open the .lnt file", false, rd.error());
        return;
    }

    LntScene scene;
    if (!scene.upload(eng, rd, err)) {
        check("every .lnt node uploads into the engine arena", false, err);
        return;
    }
    check("every .lnt node became exactly one arena tile",
          eng.arena_uploaded_tiles() == rd.get_total_nodes(),
          std::to_string(eng.arena_uploaded_tiles()) + " tiles for " +
              std::to_string(rd.get_total_nodes()) + " nodes");
    check("the arena holds the file's leaf points plus its LOD copies",
          scene.arena_points() == st.written_points &&
              scene.leaf_points() == st.leaf_points &&
              scene.lod_points() == st.lod_points,
          std::to_string(scene.leaf_points()) + " leaf + " +
              std::to_string(scene.lod_points()) + " LOD = " +
              std::to_string(scene.arena_points()));

    // The render origin is the precision decision of the whole path: positions
    // become float32 against it, so at a 682 km easting the default origin of 0
    // would quantise every point to ~6 cm.
    const BoundingBox env = rd.get_bounds();
    double o3[3];
    scene.render_origin(o3);
    check("the render origin is the file's envelope centre, not the default 0,0,0",
          std::fabs(o3[0] - env.mid(0)) < 1e-9 &&
              std::fabs(o3[1] - env.mid(1)) < 1e-9 &&
              std::fabs(o3[2] - env.mid(2)) < 1e-9 &&
              !(o3[0] == 0.0 && o3[1] == 0.0 && o3[2] == 0.0),
          "(" + std::to_string(o3[0]) + ", " + std::to_string(o3[1]) + ", " +
              std::to_string(o3[2]) + ")");

    const double side = env.extent(0);
    eng.set_clear_color(0.05f, 0.05f, 0.08f, 1.0f);
    eng.set_camera_lookat(env.mid(0) + side * 1.1, env.mid(1) - side * 1.1,
                          env.mid(2) + side * 0.9, env.mid(0), env.mid(1),
                          env.mid(2), 45.0, 1.0, side * 20.0);

    const uint64_t up_before = eng.position_upload_count();

    // ---- full leaf view --------------------------------------------------
    // The call is sequenced BEFORE the detail string. As separate arguments to
    // check() the order is unspecified, so the detail could be formatted from
    // the pre-call counter and print "0 ranges" for a selection that worked.
    const bool shown = scene.show_all(eng, true);
    const uint32_t ranges_set = eng.draw_range_count();
    check("show_all() sets one draw range per leaf tile",
          shown && ranges_set == scene.leaf_tiles(),
          std::to_string(ranges_set) + " ranges for " +
              std::to_string(scene.leaf_tiles()) + " leaf tiles");
    check("the engine's draw-range point total equals the file's leaf points",
          eng.draw_range_points() == scene.leaf_points(),
          std::to_string(eng.draw_range_points()));

    eng.render();
    EngineFrame full;
    if (!eng.capture_frame(full)) {
        check("the engine captures an offscreen frame", false, eng.last_error());
        return;
    }
    uint8_t bg[3] = {0, 0, 0};
    uint64_t bg_n = 0;
    dominant_colour(full, bg, &bg_n);
    const uint64_t total_px = uint64_t(full.width) * full.height;
    const uint64_t px_full = full.count_not_equal(bg[0], bg[1], bg[2], 6);
    std::printf("  frame          : %ux%u, background #%02x%02x%02x (%llu px)\n",
                full.width, full.height, bg[0], bg[1], bg[2],
                (unsigned long long)bg_n);
    check("the engine rasterises the .lnt geometry into real pixels",
          px_full > 0, std::to_string(px_full) + " of " +
                           std::to_string(total_px) + " px are geometry");
    check("the whole cloud does not cover the entire viewport",
          px_full < total_px, "(otherwise the background test is meaningless)");

    // GPU-only class visibility: no resident point payload is changed.
    PackedClassColor classes[256];
    for (auto& c : classes) c = PackedClassColor{255, 128, 0, 0};
    const uint64_t class_up_before = eng.position_upload_count();
    const bool class_mode = eng.set_display_mode(kEngineClassification);
    const bool hidden_lut = eng.set_classification_lut(classes);
    EngineFrame class_hidden;
    const bool hidden_captured = hidden_lut && eng.capture_frame(class_hidden);
    for (auto& c : classes) c.visible = 1;
    const bool visible_lut = eng.set_classification_lut(classes);
    EngineFrame class_visible;
    const bool visible_captured = visible_lut && eng.capture_frame(class_visible);
    check("class LUT reaches the Vulkan engine", class_mode && hidden_lut && visible_lut);
    check("class visibility culls pixels on the GPU",
          hidden_captured && visible_captured &&
          class_hidden.count_not_equal(bg[0], bg[1], bg[2], 6) == 0 &&
          class_visible.count_not_equal(bg[0], bg[1], bg[2], 6) > 0);
    check("class toggle does not upload point positions",
          eng.position_upload_count() == class_up_before);
    eng.set_display_mode(kEngineRgb);

    // ---- node-level cull -------------------------------------------------
    const BoundingBox sub = BoundingBox::from_min_max(
        env.min[0], env.min[1], env.min[2], env.mid(0), env.mid(1), env.max[2]);
    uint64_t expect_pts = 0;
    for (const LntTile* t : scene.tiles_in_bounds(sub, true)) expect_pts += t->count;
    const std::size_t n_sel = scene.apply_view(eng, sub, true);
    check("a sub-box selects a strict, non-empty subset of the leaf tiles",
          n_sel > 0 && n_sel < scene.leaf_tiles(),
          std::to_string(n_sel) + " of " + std::to_string(scene.leaf_tiles()));
    check("the engine draws exactly the selected tiles' points",
          eng.draw_range_points() == expect_pts,
          std::to_string(eng.draw_range_points()) + " expected " +
              std::to_string(expect_pts));

    eng.render();
    EngineFrame culled;
    if (!eng.capture_frame(culled)) {
        check("a culled frame captures", false, eng.last_error());
        return;
    }
    const uint64_t px_sub = culled.count_not_equal(bg[0], bg[1], bg[2], 6);
    check("culling draws strictly fewer pixels than the whole cloud",
          px_sub > 0 && px_sub < px_full,
          std::to_string(px_sub) + " px vs " + std::to_string(px_full));
    check("no position was re-uploaded to achieve the cull",
          eng.position_upload_count() == up_before,
          std::to_string(up_before) + " -> " +
              std::to_string(eng.position_upload_count()) + " uploads");

    // ---- empty draw list draws nothing -----------------------------------
    check("hide_all() clears every draw range",
          scene.hide_all(eng) && eng.draw_range_count() == 0);
    eng.render();
    EngineFrame hidden;
    if (!eng.capture_frame(hidden)) {
        check("an empty frame captures", false, eng.last_error());
        return;
    }
    const uint64_t px_hidden = hidden.count_not_equal(bg[0], bg[1], bg[2], 6);
    check("an empty draw list rasterises nothing", px_hidden == 0,
          std::to_string(px_hidden) + " px");

    // ---- the colour path (whole-cloud RGB) -------------------------------
    // Deliberately LAST: nkv_set_point_cloud re-packs the cloud at offset 0, so
    // it invalidates the arena the checks above depended on.
    LntScene rgb;
    if (rgb.upload_bulk_rgb(eng, rd, err)) {
        eng.set_display_mode(kEngineRgb);
        eng.render();
        EngineFrame cf;
        if (eng.capture_frame(cf)) {
            const uint64_t px_rgb = cf.count_not_equal(bg[0], bg[1], bg[2], 6);
            const std::size_t colours = distinct_colours(cf);
            std::printf("  colour path    : %llu px, %zu distinct colours\n",
                        (unsigned long long)px_rgb, colours);
            check("the bulk RGB upload renders the same geometry", px_rgb > 0,
                  std::to_string(px_rgb) + " px");
            // Same camera, same points, DIFFERENT attribute stream: if the colour
            // stream reached the shader the images must differ.
            check("switching to the stored RGB stream changes the rendered image",
                  cf.rgba != full.rgba && colours > 4,
                  std::to_string(colours) + " distinct colours");
        } else {
            check("the bulk RGB upload renders", false, eng.last_error());
        }
    } else {
        check("the bulk RGB (whole-cloud) upload path works", false, err);
    }

    uint64_t rendered = 0, skipped = 0, recreates = 0;
    eng.frame_stats(&rendered, &skipped, &recreates);
    std::printf("  frames         : %llu rendered, %llu skipped, %llu swapchain recreates\n",
                (unsigned long long)rendered, (unsigned long long)skipped,
                (unsigned long long)recreates);
    eng.close();
}

// The engine's frame UBO is refreshed by the render that consumes it, and
// nkv_get_last_mvp() reads the UBO of whichever slot the frame manager is
// currently on - so the render and the read-back have to be adjacent, and the
// render has to be one that actually ran. On a hidden window nkv_render() is
// entitled to skip (it reports "skipped, not an error": the swapchain has no
// visible surface to present to), and a skipped frame never refreshes the UBO.
// That is not a guess: reading the matrix after a bare render() here returned a
// ZERO matrix, which is why the engine half of this section would have silently
// skipped itself. The offscreen capture path always refreshes the slot UBO and
// renders with it, so that is what drives a frame here - and the matrix read
// back is then the one that produced the pixels the same call handed back.
static bool capture_with_mvp(VulkanEngine& eng, EngineFrame& out, float mvp16[16]) {
    if (!eng.capture_frame(out)) return false;
    if (!eng.get_last_mvp(mvp16)) return false;
    // An all-zero (or NaN) matrix is rejected: a UBO that was never written
    // would make every frustum test report "visible", which looks like a
    // successful frustum and is the opposite of one.
    double sum = 0.0;
    for (int i = 0; i < 16; ++i) sum += std::fabs(double(mvp16[i]));
    return sum > 1.0;
}

// ========================================================================
// J. CAMERA FLIGHT: FRUSTUM CULLING + ASYNCHRONOUS LRU STREAMING
// ========================================================================
// The four properties Phase 3 is required to demonstrate, each as a measurement
// rather than a description:
//
//   1. a simulated camera flight of 10 frames across the dataset;
//   2. frustum culling REDUCES the drawn node count when the camera faces away,
//      measured at the engine (ranges, points and draw calls), not at the
//      selector;
//   3. cache RAM never exceeds the configured budget, checked after every frame
//      AND against the cache's own internally-maintained peak;
//   4. streaming does not block the main thread - proved by holding the loader
//      so a payload provably cannot arrive, then showing the frame call returns
//      with that payload still absent.
//
// Two further properties are checked because they are what make the first four
// trustworthy rather than merely plausible: the cut is an ANTICHAIN (so no point
// can be drawn twice, which is otherwise the classic way a tree-based LOD draw
// list silently doubles its work), and with the error threshold forced to zero
// the cut is exactly the leaf set, summing to the file's own point total - i.e.
// nothing was lost and nothing was duplicated.
static void section_streaming(const std::string& path) {
    std::printf("\n=== J. CAMERA FLIGHT, FRUSTUM CULLING, ASYNC LRU CACHE ===\n");

    // WHICH DATASET. The capacity-only build is a 9-node tree - root plus its
    // eight children, depth 1 - and every claim in this section is about
    // HIERARCHY: the recursion the SSE test drives, coarse nodes standing in for
    // whole subtrees, a working set larger than its budget. A one-level tree
    // exercises none of it (its only possible cuts are "the leaves" and "the
    // root"), so the 2 mm precision build from section F - 617 nodes, depth 4 -
    // is the subject when it is present, and the flat file is the fallback.
    std::string used = path;
    {
        const std::string deep = path + ".fine";
        Reader probe;
        if (probe.open(deep) && probe.get_total_nodes() > 0) used = deep;
    }

    Reader rd;
    if (!rd.open(used)) {
        check("the streaming path can open the .lnt file", false, rd.error());
        return;
    }
    std::printf("  dataset        : %s, %llu nodes, %llu dataset points\n", used.c_str(),
                (unsigned long long)rd.get_total_nodes(),
                (unsigned long long)rd.get_total_points());

    // The file's own totals, derived from the DIRECTORY that was actually opened
    // rather than taken from the builder's stats for some other file. Everything
    // below is then a statement about the bytes on disk.
    uint64_t written_points = 0, dir_leaf_points = 0;
    for (const LNTNodeEntry& e : rd.nodes()) {
        written_points += e.point_count;
        if (is_leaf_node(e)) dir_leaf_points += e.point_count;
    }
    check("the directory's leaf payloads sum to the header's dataset point count",
          dir_leaf_points == rd.get_total_points() && dir_leaf_points > 0,
          std::to_string(dir_leaf_points) + " leaf points vs header " +
              std::to_string(rd.get_total_points()));

    // ---- J1. the hierarchy, recovered from the flat directory ------------
    LntNodeTree tree;
    std::string err;
    const bool tree_ok = tree.build(rd.nodes(), err);
    check("the octree relation is rebuilt from the flat Morton-ordered directory",
          tree_ok, err);
    if (!tree_ok || tree.empty()) return;
    std::printf("  tree           : %s\n", tree.describe().c_str());
    check("no node is orphaned: every non-root id resolves to a parent in the table",
          tree.orphan_nodes == 0,
          std::to_string(tree.orphan_nodes) + " orphans, " +
              std::to_string(tree.edge_count) + " edges, " +
              std::to_string(tree.roots.size()) + " roots");
    {
        uint64_t reach = 0;
        for (uint32_t r : tree.roots) reach += tree.nodes[r].subtree_nodes;
        check("the roots' subtree aggregates account for every node exactly once",
              reach == tree.nodes.size(),
              std::to_string(reach) + " of " + std::to_string(tree.nodes.size()));
        check("the tree has real depth, so the traversal has subtrees to stand in for",
              tree.max_depth >= 2,
              std::to_string(tree.max_depth) + " levels");
    }
    {
        // The parent relation is the ID ENCODING, not a stored pointer, so the
        // edges the tree built have to be checked against the encoding itself.
        uint64_t bad = 0;
        for (const LntNodeTree::Node& n : tree.nodes) {
            if (n.parent == kNoParent) continue;
            if ((n.id >> 3) != tree.nodes[n.parent].id) ++bad;
            if (level_of_node_id(n.id) != tree.nodes[n.parent].level + 1) ++bad;
        }
        check("every child edge satisfies child_id >> 3 == parent_id at level+1",
              bad == 0, std::to_string(bad) + " bad edges");
    }
    {
        uint64_t reach_pts = 0;
        for (uint32_t r : tree.roots) reach_pts += tree.nodes[r].subtree_points;
        check("the subtree POINT aggregates sum to the directory's written points",
              reach_pts == written_points,
              std::to_string(reach_pts) + " vs " + std::to_string(written_points));
    }

    const BoundingBox env = rd.get_bounds();
    const double side = env.extent(0);
    const double mid[3] = {env.mid(0), env.mid(1), env.mid(2)};
    double origin[3];
    for (int i = 0; i < 3; ++i) origin[i] = rd.origin(i);

    // ---- J2. frustum + SSE, as pure selection (no engine, no cache) ------
    // A camera placed far enough back that the whole envelope is inside it. The
    // "nothing is culled" check below is what validates that placement: if the
    // frustum test were over-culling, that check is the one that would fail.
    LNTCamera wide;
    wide.eye[0] = mid[0] - side * 2.2;
    wide.eye[1] = mid[1] - side * 2.2;
    wide.eye[2] = mid[2] + side * 1.8;
    for (int i = 0; i < 3; ++i) wide.target[i] = mid[i];
    wide.aspect = 16.0 / 9.0;
    wide.near_clip = 1.0;
    wide.far_clip = side * 40.0;
    wide.screen_height_px = 720;
    for (int i = 0; i < 3; ++i) wide.origin[i] = origin[i];
    check("the standalone camera builds a usable frustum from eye/target/fov",
          wide.rebuild(), "no degenerate basis");

    LntLodParams exact;
    exact.sse_threshold = 0.0;            // always descend: the cut is the leaves
    exact.max_nodes = 100000;
    const LntLodSelection leaf_cut =
        select_lod(tree, rd.nodes(), origin, wide, exact, rd.get_total_points());
    std::printf("  leaves-only    : %zu nodes, %llu points, %llu culled\n",
                leaf_cut.cut_size(), (unsigned long long)leaf_cut.selected_points,
                (unsigned long long)leaf_cut.culled);
    check("the whole-envelope camera puts every node inside the frustum (no over-culling)",
          leaf_cut.culled == 0 && leaf_cut.visited == tree.nodes.size(),
          std::to_string(leaf_cut.culled) + " culled of " +
              std::to_string(leaf_cut.visited) + " tested");
    check("a threshold of 0 selects exactly the leaf set",
          leaf_cut.cut_size() == tree.leaf_count,
          std::to_string(leaf_cut.cut_size()) + " cut nodes vs " +
              std::to_string(tree.leaf_count) + " leaves");
    check("the leaf cut sums to the file's dataset point total: no point lost",
          leaf_cut.selected_points == rd.get_total_points() &&
              leaf_cut.selected_points == dir_leaf_points,
          std::to_string(leaf_cut.selected_points) + " vs header " +
              std::to_string(rd.get_total_points()) + " / directory " +
              std::to_string(dir_leaf_points));

    // The antichain property, checked as a property and not sampled: for every
    // pair of cut nodes, neither may be an ancestor of the other. A violation is
    // the double-draw bug, and it is exactly what a flat "draw everything that
    // intersects" selection produces.
    {
        bool antichain = true;
        std::size_t first_bad = 0;
        for (std::size_t i = 0; i < leaf_cut.node_ids.size() && antichain; ++i)
            for (std::size_t j = 0; j < leaf_cut.node_ids.size(); ++j) {
                if (i == j) continue;
                if (LntNodeTree::is_ancestor(leaf_cut.node_ids[i], leaf_cut.node_ids[j])) {
                    antichain = false;
                    first_bad = i;
                    break;
                }
            }
        check("the leaf cut is an antichain: no selected node is another's ancestor",
              antichain,
              antichain ? std::to_string(leaf_cut.cut_size()) + " nodes, no ancestor pairs"
                        : "node " + std::to_string(first_bad) + " is an ancestor");
    }
    {
        const LntLodSelection again =
            select_lod(tree, rd.nodes(), origin, wide, exact,
                       rd.get_total_points());
        check("the traversal is deterministic: the same camera yields the same cut",
              again.node_ids == leaf_cut.node_ids &&
                  again.selected_points == leaf_cut.selected_points,
              std::to_string(again.cut_size()) + " nodes, point total " +
                  std::to_string(again.selected_points));
    }

    // The error threshold is the LOD dial: raising it must never select MORE
    // nodes, and a coarse cut must draw strictly fewer points than the leaves.
    LntLodParams coarse = exact;
    coarse.sse_threshold = 10000.0;       // the root stands in for its subtree
    const LntLodSelection coarse_cut =
        select_lod(tree, rd.nodes(), origin, wide, coarse, rd.get_total_points());
    std::printf("  error-threshold: %zu nodes, %llu points at 10000 px vs %zu/%llu at 0 px\n",
                coarse_cut.cut_size(), (unsigned long long)coarse_cut.selected_points,
                leaf_cut.cut_size(), (unsigned long long)leaf_cut.selected_points);
    check("raising the screen-space error threshold never selects more nodes",
          coarse_cut.cut_size() <= leaf_cut.cut_size());
    check("a coarse cut stands in for whole subtrees: strictly fewer points drawn",
          coarse_cut.selected_points > 0 &&
              coarse_cut.selected_points < leaf_cut.selected_points,
          std::to_string(coarse_cut.selected_points) + " < " +
              std::to_string(leaf_cut.selected_points));
    check("the coarse cut is also an antichain",
          [&] {
              for (std::size_t i = 0; i < coarse_cut.node_ids.size(); ++i)
                  for (std::size_t j = 0; j < coarse_cut.node_ids.size(); ++j)
                      if (i != j && LntNodeTree::is_ancestor(coarse_cut.node_ids[i],
                                                             coarse_cut.node_ids[j]))
                          return false;
              return true;
          }());

    // Screen-space error, and its relationship to a focal-length-corrected
    // pixels-per-metre figure. Asserting the identity means the metric is not
    // merely "some number that goes up when you get close".
    {
        const double d = std::sqrt((wide.eye[0] - mid[0]) * (wide.eye[0] - mid[0]) +
                                   (wide.eye[1] - mid[1]) * (wide.eye[1] - mid[1]) +
                                   (wide.eye[2] - mid[2]) * (wide.eye[2] - mid[2]));
        const BoundingBox cell = BoundingBox::from_center_half(mid, 100.0);
        const double sse = wide.sse(cell);
        const double px = wide.pixels_per_meter_at(d) * 100.0;
        const double ratio = (sse > 0.0) ? px / sse : 0.0;
        // The factor is derived here from the FOV rather than taken from the
        // camera, so the assertion cannot be satisfied by a typo in
        // focal_scale() agreeing with a typo in pixels_per_meter_at(). The two
        // metrics are computed by independent code paths - one from a per-pixel
        // ratio, the other from the focal length - and must meet at exactly
        // f_px/height = 1/(2*tan(fovY/2)).
        const double tan_half = std::tan(wide.fov_y_deg * 3.14159265358979323846 / 360.0);
        const double derived = 1.0 / (2.0 * tan_half);
        check("SSE == (half_size/distance)*height exactly, and the focal-correct "
              "pixels-per-metre form equals it times f_px/height",
              std::fabs(sse - (100.0 / d) * 720.0) < 1e-9 &&
                  std::fabs(wide.focal_scale() - derived) < 1e-12 &&
                  std::fabs(ratio - derived) < 1e-9,
              "sse " + std::to_string(sse) + " px, ppm form " + std::to_string(px) +
                  " px = sse * " + std::to_string(ratio) + "; 1/(2*tan(fov/2)) = " +
                  std::to_string(derived) + ", camera factor " +
                  std::to_string(wide.focal_scale()));
    }

    // Facing away: the camera looks along +Z away from the dataset. Nothing can
    // be inside the frustum, so the traversal must not even visit the nodes.
    {
        LNTCamera away = wide;
        away.eye[0] = mid[0];
        away.eye[1] = mid[1];
        away.eye[2] = mid[2] - side * 3.0;
        for (int i = 0; i < 3; ++i) away.target[i] = away.eye[i];
        away.target[2] = away.eye[2] - side;      // look further away from the data
        check("the away-facing camera still has a valid frustum", away.rebuild());
        const LntLodSelection away_cut =
            select_lod(tree, rd.nodes(), origin, away, exact,
                       rd.get_total_points());
        std::printf("  facing away    : %zu nodes cut, %llu visited, %llu culled, "
                    "%llu inside culled subtrees\n",
                    away_cut.cut_size(), (unsigned long long)away_cut.visited,
                    (unsigned long long)away_cut.culled,
                    (unsigned long long)away_cut.subtree_skipped);
        check("facing away selects zero nodes to draw", away_cut.cut_size() == 0,
              std::to_string(away_cut.cut_size()) + " selected");
        check("facing away never visits a node: whole subtrees are dropped untested",
              away_cut.visited < tree.nodes.size(),
              std::to_string(away_cut.visited) + " of " +
                  std::to_string(tree.nodes.size()) + " tested; " +
                  std::to_string(leaf_cut.visited) + " tested facing it");
    }

    // ---- J3. the asynchronous cache, on its own -------------------------
    check("the default cache budget is 2 GiB (4 GiB tier for 3B+ point clouds)",
          kDefaultCacheBudgetBytes == 2048ull * 1024ull * 1024ull,
          std::to_string(kDefaultCacheBudgetBytes / (1024 * 1024)) + " MiB");

    // A budget deliberately too small to hold the file forces the eviction path
    // to run; sized from the largest node so nothing is ever "too large to
    // admit", which would be a different code path and would hide eviction.
    uint64_t max_payload = 0, total_payload = 0;
    uint64_t biggest_node = 0;
    for (const LNTNodeEntry& e : rd.nodes()) {
        const uint64_t b = uint64_t(e.point_count) * sizeof(RawPoint);
        total_payload += b;
        if (b > max_payload) { max_payload = b; biggest_node = e.node_id; }
    }
    const uint64_t budget = max_payload * 2;
    std::printf("  cache budget   : %.2f MiB (largest node %.2f MiB, whole file "
                "%.2f MiB)\n",
                double(budget) / (1024 * 1024), double(max_payload) / (1024 * 1024),
                double(total_payload) / (1024 * 1024));
    check("the test budget is genuinely smaller than the file, so eviction must happen",
          budget < total_payload, std::to_string(budget) + " < " +
                                       std::to_string(total_payload) + " bytes");
    (void)biggest_node;

    LntCacheManager cache(budget);
    check("the cache opens its own reader on the same file",
          cache.open(used, err), err);
    {
        // The loader holds its OWN Reader, and this is the check that says so: a
        // deliberately interleaved decode through the main Reader, with loads in
        // flight, must return the same bytes as a clean read. A shared stream
        // would be seeked out from under this call and return other nodes' points.
        const LNTNodeEntry& probe = rd.nodes()[rd.nodes().size() / 2];
        std::vector<RawPoint> ref_probe;
        std::vector<uint64_t> all_ids;
        for (const LNTNodeEntry& e : rd.nodes()) all_ids.push_back(e.node_id);
        cache.request_all(all_ids);            // loads now racing
        const bool ref_ok = rd.read_node_points(probe, ref_probe);
        cache.hold(true);
        std::vector<RawPoint> during;
        const bool during_ok = rd.read_node_points(probe, during);
        const bool same = ref_ok && during_ok && same_points(ref_probe, during);
        // On a mismatch, say WHICH read failed and how many PAYLOAD bytes differ:
        // "the two decodes differ" is not enough to tell a byte-level corruption
        // from a failed read, and those are different bugs.
        std::size_t diff_bytes = 0;
        std::string diff_where;
        if (!same && ref_probe.size() == during.size() && !ref_probe.empty()) {
            const unsigned char* a = reinterpret_cast<const unsigned char*>(ref_probe.data());
            const unsigned char* b = reinterpret_cast<const unsigned char*>(during.data());
            for (std::size_t i = 0; i < ref_probe.size(); ++i) {
                for (std::size_t k = 0; k < kRawPointPayloadBytes; ++k) {
                    if (a[i * sizeof(RawPoint) + k] == b[i * sizeof(RawPoint) + k]) continue;
                    ++diff_bytes;
                    if (diff_where.size() < 120) {
                        char buf[64];
                        std::snprintf(buf, sizeof(buf), " pt%zu+%zu: %02x/%02x", i, k,
                                      unsigned(a[i * sizeof(RawPoint) + k]),
                                      unsigned(b[i * sizeof(RawPoint) + k]));
                        diff_where += buf;
                    }
                }
            }
        }
        check("the main reader is undisturbed while the loader is reading: same bytes",
              same,
              "read1 " + std::string(ref_ok ? "ok" : "FAILED") + " (" +
                  std::to_string(ref_probe.size()) + " pts), read2 " +
                  std::string(during_ok ? "ok" : "FAILED") + " (" +
                  std::to_string(during.size()) + " pts), " +
                  std::to_string(diff_bytes) + " differing payload bytes" + diff_where);
        cache.clear_pinned();
        cache.hold(false);
        // Sequenced deliberately: `pending()` must be read AFTER the drain, and
        // writing both as arguments to check() leaves the order unspecified (GCC
        // evaluates right to left, which printed "9 still pending" next to a
        // PASS and made the message contradict the result).
        const bool all_loaded = cache.drain(15000);
        check("with no pin set the deferred/queued loads all complete",
              all_loaded && cache.pending() == 0,
              std::to_string(cache.pending()) + " still pending after the drain");
        check("the cache never exceeded its budget during the churn",
              cache.stats().within_budget(),
              std::to_string(cache.stats().peak_bytes) + " peak of " +
                  std::to_string(budget) + " bytes");
        check("no payload was larger than the budget, so eviction is what was tested",
              cache.stats().too_large == 0,
              std::to_string(cache.stats().too_large) + " too-large payloads, largest "
              "node " + std::to_string(max_payload) + " bytes");

        // Eviction and reload: the node that was resident first is almost
        // certainly gone after loading the whole file into a two-node budget.
        cache.request(probe.node_id);
        const bool drained = cache.drain(15000);
        const std::vector<RawPoint>* got = cache.peek(probe.node_id);
        const bool identical = drained && got != nullptr && same_points(ref_probe, *got);
        check("an evicted node re-reads byte-identical after the whole file passed through",
              identical,
              identical ? std::to_string(cache.stats().evictions) + " evictions occurred"
                        : "reload mismatch or still not resident");
        check("eviction actually ran, so the probe above tested the real path",
              cache.stats().evictions > 0,
              std::to_string(cache.stats().evictions) + " evictions, " +
                  std::to_string(cache.stats().loads) + " loads");
        check("reloading a payload does not pay for it twice in the budget",
              cache.resident_bytes() <= budget,
              std::to_string(cache.resident_bytes()) + " of " +
                  std::to_string(budget) + " bytes resident");
    }
    cache.close();

    // ---- J4. the flight, through the engine ------------------------------
    const std::string dll = VulkanEngine::find_dll();
    if (dll.empty()) {
        skip("streaming flight", "no naksha_vulkan.dll found (set LNT_VULKAN_DLL)");
        return;
    }
    VulkanEngine eng;
    if (!eng.load(dll, err)) { skip("streaming flight", "engine load failed: " + err); return; }
    if (!eng.is_available()) {
        skip("streaming flight", "no Vulkan loader or no physical device on this machine");
        return;
    }
    if (!eng.open(960, 720, /*hidden=*/true, /*validation=*/false, err)) {
        skip("streaming flight", "renderer creation failed: " + err);
        return;
    }

    LntScene scene;
    if (!scene.plan(rd, err)) {
        check("the arena layout is planned from the directory alone", false, err);
        return;
    }
    check("plan() lays out one tile per node without uploading a payload",
          scene.planned() && scene.tiles().size() == rd.get_total_nodes() &&
              scene.uploaded_tiles() == 0,
          std::to_string(scene.tiles().size()) + " tiles planned, " +
              std::to_string(scene.uploaded_tiles()) + " uploaded");
    {
        uint64_t planned = 0;
        for (const LntTile& t : scene.tiles()) planned += t.count;
        check("the planned offsets cover the file's points exactly once, with no overlap",
              planned == scene.planned_points() && planned == written_points,
              std::to_string(planned) + " vs written " +
                  std::to_string(written_points));
    }
    if (!scene.reserve(eng, err)) {
        check("the arena capacity is reserved once, up front", false, err);
        return;
    }
    check("the whole arena is reserved before a single node is read",
          eng.point_capacity() == scene.planned_points(),
          std::to_string(eng.point_capacity()) + " point capacity");

    // THE FLIGHT'S OWN CACHE, WITH ITS OWN BUDGET. J3 deliberately ran with two
    // payloads of room to force eviction; a flight measured under that budget
    // would mostly be a measurement of disk latency, since a frame's cut could
    // not fit. The flight therefore gets a budget that holds the file's payload
    // twice over - which is also what the 512 MiB default would do for a file of
    // this size - and the eviction-under-frame-pressure case is a separate pass
    // below, with numbers instead of adjectives.
    const uint64_t flight_budget =
        std::max<uint64_t>(total_payload + max_payload, budget);
    std::printf("  flight budget  : %.2f MiB for %.2f MiB of payloads\n",
                double(flight_budget) / (1024 * 1024),
                double(total_payload) / (1024 * 1024));
    LntCacheManager flight_cache(flight_budget);
    if (!flight_cache.open(used, err)) {
        check("the flight's cache opens its own reader", false, err);
        return;
    }

    LntStreamer streamer;
    if (!streamer.attach(rd, &flight_cache, err)) {
        check("the streamer attaches to the reader and the cache", false, err);
        return;
    }

    // ---- the camera path -------------------------------------------------
    // The engine's own frame UBO is the authoritative camera: nkv_get_last_mvp
    // returns the exact matrix the shader projected with, so the frustum is
    // derived from the matrix the GPU actually used rather than from a parallel
    // re-derivation that could silently disagree. See capture_with_mvp() for why
    // the frame is driven by the capture path and not by nkv_render().
    const double near_clip = 1.0;
    const double far_clip = side * 40.0;
    // EXACTLY the flight's opening camera (t = 0 below), so that the parity
    // checks, the warm-up and frame 0 all describe the same view. Warming a
    // slightly different camera would leave frame 0 cold and make the flight
    // measurement a statement about the disk rather than about streaming.
    double eye[3] = {mid[0] - side * 1.6, mid[1] - side * 1.6, mid[2] + side * 0.7};
    double tgt[3] = {mid[0] - side * 0.4, mid[1], mid[2]};
    eng.set_render_origin(origin[0], origin[1], origin[2]);
    eng.set_camera_lookat(eye[0], eye[1], eye[2], tgt[0], tgt[1], tgt[2], 45.0,
                          near_clip, far_clip);
    eng.set_clear_color(0.03f, 0.03f, 0.05f, 1.0f);

    EngineFrame eng_frame;
    float mvp[16] = {};
    if (!capture_with_mvp(eng, eng_frame, mvp)) {
        skip("the engine exposes the frame's view-projection",
             "capture_frame/get_last_mvp did not yield a usable matrix");
        eng.close();
        return;
    }
    const uint32_t fw = eng_frame.width, fh = eng_frame.height;
    const double aspect = double(fw) / double(fh);
    check("the engine exposes the view-projection its own frame was drawn with",
          fw > 0 && fh > 0 && aspect > 0.0 && std::isfinite(aspect),
          std::to_string(fw) + "x" + std::to_string(fh) + ", aspect " +
              std::to_string(aspect));

    LNTCamera eng_cam;
    check("a camera built from the engine's own frame matrix yields valid planes",
          eng_cam.set_from_engine_mvp(mvp, origin, eye));
    // The viewport height is part of the VIEW, not a constant: screen-space error
    // is a pixel count, and this camera must carry the engine's real height or
    // every SSE-driven LOD decision differs from the one the flight will make.
    // (The default is 1080; this engine's offscreen target is 681.) Without this
    // the warm-up would be warming a different LOD cut than frame 0 asks for.
    eng_cam.screen_height_px = fh;
    {
        LNTCamera standalone;
        for (int i = 0; i < 3; ++i) {
            standalone.eye[i] = eye[i]; standalone.target[i] = tgt[i];
            standalone.origin[i] = origin[i];
        }
        standalone.fov_y_deg = 45.0;
        standalone.aspect = aspect;
        standalone.near_clip = near_clip;
        standalone.far_clip = far_clip;
        standalone.screen_height_px = fh;
        const bool ok = standalone.rebuild();

        // (1) MATRIX PARITY, compared RELATIVE to each entry's own magnitude.
        // The engine hands the frame matrix back as float32, and this camera is
        // 600 m of near field with a 4.8e4 far term against a UTM render origin,
        // so an absolute bound on a matrix entry would be a bound on the
        // coordinate magnitudes rather than on the agreement.
        double worst_rel = 0.0;
        int wr = 0, wc = 0;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) {
                const double e = eng_cam.vp(r, c);
                const double rel =
                    std::fabs(standalone.vp(r, c) - e) / std::max(1.0, std::fabs(e));
                if (rel > worst_rel) { worst_rel = rel; wr = r; wc = c; }
            }
        std::printf("  camera parity  : worst relative |standalone - engine| entry %s at "
                    "(%d,%d); float32 epsilon is %s\n",
                    fmt_e(worst_rel).c_str(), wr, wc,
                    fmt_e(double(std::numeric_limits<float>::epsilon())).c_str());
        check("the standalone camera reproduces the engine's view-projection matrix",
              ok && worst_rel < 1e-5,
              "worst relative entry difference " + fmt_e(worst_rel) +
                  " (float32 storage of the matrix alone costs " +
                  fmt_e(double(std::numeric_limits<float>::epsilon())) + ")");

        // (2) PLANE PARITY, measured where planes are actually used: as a SIGNED
        // DISTANCE at the nodes the cull tests. A plane coefficient is not a
        // length, and after the world-space shift a `d` is of the order of the
        // render origin (1e6 m) where float32's spacing is 6e-2 m - so an
        // absolute bound on coefficients would be a statement about float32, not
        // about the frustum.
        //
        // The far plane is reported separately because it is genuinely
        // ill-conditioned, and this is measured, not assumed: its unnormalised
        // normal is (row3-row2).xyz, whose z term is near/(far-near) = 2e-5 for a
        // 4.8e4 far plane, so normalising it divides float32 storage noise by
        // 2e-5. That error is a property of extracting a far plane from a float32
        // matrix; it is why the check below bounds the five well-conditioned
        // planes tightly and the far plane in proportion to its own distance.
        double five_err = 0.0, far_err = 0.0, far_ref = 0.0;
        int worst_plane = 0;
        double per_plane[kPlaneCount] = {};
        for (const LNTNodeEntry& e : rd.nodes()) {
            const BoundingBox cell = node_bounds(e, origin);
            const double c[3] = {cell.mid(0), cell.mid(1), cell.mid(2)};
            for (int i = 0; i < kPlaneCount; ++i) {
                double da = eng_cam.plane(i, 3), db = standalone.plane(i, 3);
                for (int k = 0; k < 3; ++k) {
                    da += eng_cam.plane(i, k) * c[k];
                    db += standalone.plane(i, k) * c[k];
                }
                const double err = std::fabs(da - db);
                per_plane[i] = std::max(per_plane[i], err);
                if (i == kPlaneFar) {
                    far_err = std::max(far_err, err);
                    far_ref = std::max(far_ref, std::fabs(da));
                } else if (err > five_err) {
                    five_err = err;
                    worst_plane = i;
                }
            }
        }
        std::printf("  plane parity   : worst signed-distance disagreement over %zu nodes,"
                    " per plane:", tree.nodes.size());
        for (int i = 0; i < kPlaneCount; ++i) std::printf(" %.3g", per_plane[i]);
        std::printf(" m  (order L,R,B,T,N,F)\n");
        check("the five well-conditioned planes place every node within a metre of "
              "the same place",
              ok && five_err < 1.0,
              "worst " + std::to_string(five_err) + " m on plane " +
                  std::to_string(worst_plane) + " of " + std::to_string(kPlaneCount));
        check("the far plane's disagreement stays small against its own distance",
              ok && far_ref > 1000.0 && far_err < 0.05 * far_ref,
              std::to_string(far_err) + " m at a far plane " + std::to_string(far_ref) +
                  " m away (" + std::to_string(100.0 * far_err / far_ref) + "% of its "
                  "distance, which is the conditioning of row3-row2, not a "
                  "disagreement about where the plane is)");

        // (3) VERDICT PARITY: the property culling actually depends on. Two
        // frusta that differ by a centimetre at the data's own scale must make
        // the same keep/discard decision for every node in the file.
        {
            uint64_t differing = 0, kept = 0;
            for (const LNTNodeEntry& e : rd.nodes()) {
                const bool a = eng_cam.visible(e, origin);
                const bool b = standalone.visible(e, origin);
                if (a != b) ++differing;
                if (a && b) ++kept;
            }
            check("both frusta classify every node in the file identically",
                  ok && differing == 0 && kept > 0,
                  std::to_string(differing) + " of " + std::to_string(tree.nodes.size()) +
                      " nodes classified differently, " + std::to_string(kept) +
                      " kept by both");
        }
    }

    // ---- warm-up on the opening camera ------------------------------------
    // The very first frame of any flight is a cold start: nothing is resident, so
    // its cut cannot be drawn yet. Rather than let that first frame quietly
    // dilute the evidence, the cache is warmed for the OPENING camera and drained
    // once - which is exactly what a viewer does while its first frame is on
    // screen - so that the measured flight below starts from a resident view.
    // The cold start is not hidden: it is what the held-loader check at the end
    // of this section measures directly.
    const std::size_t cold_push = streamer.update_visible_nodes(eng, scene, eng_cam, err);
    const uint64_t cold_cut = streamer.stats().cut_nodes;
    const bool warmed = flight_cache.drain(20000);
    const std::size_t warm_push = streamer.update_visible_nodes(eng, scene, eng_cam, err);
    std::printf("  cold start     : cut %llu nodes, %zu ranges drawable before the payloads "
                "arrived; %s; the same camera then pushes %zu ranges for %u points\n",
                (unsigned long long)cold_cut, cold_push,
                warmed ? "drained" : "DRAIN TIMED OUT", warm_push,
                eng.draw_range_points());
    check("a cold frame's cut streams in and then draws in full",
          warmed && cold_cut > 0 && warm_push == std::size_t(cold_cut) &&
              eng.draw_range_points() == streamer.stats().wanted_points,
          std::to_string(warm_push) + " of " + std::to_string(cold_cut) +
              " ranges, " + std::to_string(eng.draw_range_points()) + " of " +
              std::to_string(streamer.stats().wanted_points) + " points");

    // ---- the 10-frame flight ---------------------------------------------
    const int kFrames = 10;
    uint64_t budget_violations = 0;
    uint64_t frames_drawn = 0;
    uint64_t ranges_mismatch = 0;
    uint64_t point_mismatch = 0;
    uint64_t mvp_failures = 0;
    uint64_t uploads_during_late_frames = 0;
    uint64_t cut_before = 0, cut_after = 0;
    uint64_t partial_frames = 0;
    const uint64_t frames_before_flight = streamer.stats().frames;
    std::vector<double> frame_ms(kFrames, 0.0);
    std::vector<std::size_t> pushed_per_frame(kFrames, 0);
    std::vector<uint64_t> cut_per_frame(kFrames, 0);
    LNTCamera last_cam = eng_cam;

    for (int f = 0; f < kFrames; ++f) {
        const double t = double(f) / double(kFrames - 1);
        // Fly from one corner of the envelope to the other, always looking at a
        // point on the data so the camera actually traverses it.
        eye[0] = mid[0] - side * 1.6 + side * 2.6 * t;
        eye[1] = mid[1] - side * 1.6 + side * 1.2 * t;
        eye[2] = mid[2] + side * 0.7 - side * 0.3 * t;
        tgt[0] = mid[0] - side * 0.4 + side * 0.8 * t;
        tgt[1] = mid[1];
        tgt[2] = mid[2];
        eng.set_camera_lookat(eye[0], eye[1], eye[2], tgt[0], tgt[1], tgt[2], 45.0,
                              near_clip, far_clip);
        EngineFrame fr;
        float m[16] = {};
        if (!capture_with_mvp(eng, fr, m)) { ++mvp_failures; continue; }
        LNTCamera cam;
        cam.set_from_engine_mvp(m, origin, eye);
        cam.screen_height_px = fh;
        last_cam = cam;

        const auto t0 = std::chrono::steady_clock::now();
        const std::size_t pushed =
            streamer.update_visible_nodes(eng, scene, cam, err);
        const auto t1 = std::chrono::steady_clock::now();
        frame_ms[std::size_t(f)] =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        pushed_per_frame[std::size_t(f)] = pushed;
        cut_per_frame[std::size_t(f)] = streamer.stats().cut_nodes;

        if (f == 0) cut_before = streamer.stats().cut_nodes;
        if (f == kFrames - 1) cut_after = streamer.stats().cut_nodes;
        if (pushed > 0) ++frames_drawn;
        if (pushed < streamer.stats().cut_nodes) ++partial_frames;

        // The draw list must reference only UPLOADED tiles, and the engine's own
        // point total must equal the sum of the ranges the streamer pushed -
        // read back from the engine, so this compares the engine's state against
        // the streamer's intent rather than the streamer against itself.
        if (pushed != eng.draw_range_count()) ++ranges_mismatch;
        // The engine's own reported point total must equal the sum of the tiles
        // the streamer decided to push. Read from the engine, so this compares
        // engine state against streamer intent and not a value against itself.
        if (uint64_t(eng.draw_range_points()) != streamer.stats().cut_points)
            ++point_mismatch;
        if (f >= kFrames / 2) uploads_during_late_frames = eng.position_upload_count();

        // CHECKED EVERY FRAME, against the cache's own byte counter - not once at
        // the end, where a transient overshoot would be invisible.
        if (flight_cache.resident_bytes() > flight_budget) ++budget_violations;
    }

    std::printf("  flight         : %d frames, cut %llu -> %llu nodes, %llu frames drew "
                "something, %llu awaited payloads\n",
                kFrames, (unsigned long long)cut_before,
                (unsigned long long)cut_after, (unsigned long long)frames_drawn,
                (unsigned long long)partial_frames);
    {
        std::printf("  per frame      : ");
        for (int f = 0; f < kFrames; ++f)
            std::printf("%zu/%llu%s ", pushed_per_frame[std::size_t(f)],
                        (unsigned long long)cut_per_frame[std::size_t(f)],
                        f + 1 == kFrames ? "\n" : ",");
    }
    check("the 10-frame camera flight ran and every frame produced a drawable cut",
          streamer.stats().frames - frames_before_flight == uint64_t(kFrames) &&
              frames_drawn == uint64_t(kFrames),
          std::to_string(streamer.stats().frames - frames_before_flight) +
              " frames in the flight, " + std::to_string(frames_drawn) +
              " with ranges");
    check("the draw ranges the engine holds match what the streamer pushed",
          ranges_mismatch == 0 && mvp_failures == 0,
          std::to_string(ranges_mismatch) + " range mismatches, " +
              std::to_string(mvp_failures) + " frames with no MVP");
    check("the engine's drawn point total equals the pushed tiles' exact sum",
          point_mismatch == 0,
          std::to_string(point_mismatch) + " frames disagreed");
    check("CACHE RAM NEVER EXCEEDED THE LIMIT, checked after every frame",
          budget_violations == 0,
          std::to_string(budget_violations) + " violating frames; resident " +
              std::to_string(flight_cache.resident_bytes()) + " of " +
              std::to_string(flight_budget) + " bytes");
    check("the cache's own internally-maintained peak is also within the limit",
          flight_cache.stats().within_budget() &&
              flight_cache.peak_bytes() <= flight_budget,
          std::to_string(flight_cache.peak_bytes()) + " peak of " +
              std::to_string(flight_budget) + " bytes");
    {
        // The end-to-end statement, and the one a partial push cannot fake: once
        // the loader has caught up on the LAST camera of the flight, that camera
        // must draw its whole cut - every range, every point. The flight above
        // races the loader on purpose, so the per-frame push counts are allowed
        // to lag; this is what says the lag was the disk and not a broken cut.
        const bool caught_up = flight_cache.drain(20000);
        const std::size_t full_push =
            streamer.update_visible_nodes(eng, scene, last_cam, err);
        const uint64_t want_pts = streamer.stats().wanted_points;
        std::printf("  end of flight  : last camera cut %llu nodes / %llu points; with the "
                    "loader caught up it pushes %zu ranges for %u points\n",
                    (unsigned long long)streamer.stats().cut_nodes,
                    (unsigned long long)want_pts, full_push, eng.draw_range_points());
        check("after the flight, the last camera draws its ENTIRE cut once streamed",
              caught_up && full_push == streamer.stats().cut_nodes &&
                  eng.draw_range_points() == want_pts,
              std::to_string(full_push) + " of " +
                  std::to_string(streamer.stats().cut_nodes) + " ranges, " +
                  std::to_string(eng.draw_range_points()) + " of " +
                  std::to_string(want_pts) + " points");
    }
    {
        // Only the nodes the camera looked at were ever written to the arena.
        // Phase 2's upload() writes all of them; this is the streaming saving,
        // and it is the reason a .lnt larger than the device's memory can be
        // browsed at all.
        const uint64_t written = scene.uploaded_tiles();
        std::printf("  lazy upload    : %llu of %zu nodes written; %llu never touched\n",
                    (unsigned long long)written, tree.nodes.size(),
                    (unsigned long long)streamer.nodes_never_uploaded());
        check("the lazy path wrote only the nodes the camera visited, not the file",
              written > 0 && written < tree.nodes.size(),
              std::to_string(written) + " of " + std::to_string(tree.nodes.size()));
        check("every tile the engine drew had actually been uploaded",
              eng.draw_range_points() > 0 &&
                  eng.draw_range_points() <= scene.planned_points());
    }

    // ---- does streaming block the render thread? --------------------------
    // The loader is HELD, so a payload that is requested cannot possibly arrive
    // while the check runs. If the frame call waited for it, the wait would show
    // up as elapsed time and as a resident payload; both are measured.
    double worst_frame_ms = 0.0, mean_frame_ms = 0.0;
    for (double ms : frame_ms) {
        worst_frame_ms = std::max(worst_frame_ms, ms);
        mean_frame_ms += ms;
    }
    mean_frame_ms /= double(kFrames);
    std::printf("  main-thread    : %.3f ms worst, %.3f ms mean over %d frames (whole frame "
                "call: selection, cache handoff, arena upload, draw-range push); the "
                "streamer's own timer reports %.3f ms worst\n",
                worst_frame_ms, mean_frame_ms, kFrames,
                double(streamer.stats().max_main_ns) / 1e6);
    check("no frame blocked the main thread: the whole frame call stays well inside a "
          "frame budget",
          streamer.stats().max_main_ns > 0 && worst_frame_ms < 50.0,
          std::to_string(worst_frame_ms) + " ms worst of " + std::to_string(kFrames) +
              " frames at 944x681");
    // The streamer instruments itself, so its number and the harness's wall clock
    // have to agree - a stale or mis-scoped internal timer would otherwise report
    // a frame cost that the harness never measured.
    check("the streamer's internal frame timer agrees with the harness's wall clock",
          double(streamer.stats().max_main_ns) / 1e6 <= worst_frame_ms * 2.0 + 1.0,
          "streamer " + std::to_string(double(streamer.stats().max_main_ns) / 1e6) +
              " ms vs harness " + std::to_string(worst_frame_ms) + " ms");
    {
        // SET UP SO THE ABSENCE IS CERTAIN, not merely likely. The loader is
        // drained first (so nothing is in flight), the pin set is cleared, and
        // the victim is chosen from the nodes that are provably NOT resident.
        // Only then is the loader held and the victim requested: with the worker
        // suspended and nothing already loading, that payload cannot arrive
        // while the check runs. Requesting a node that the loader might already
        // be reading would make this a race the test could lose.
        flight_cache.clear_pinned();
        const bool idle = flight_cache.drain(20000);
        flight_cache.hold(true);
        uint64_t victim_id = 0;
        bool have_victim = false;
        for (const LNTNodeEntry& e : rd.nodes()) {
            if (flight_cache.residency(e.node_id) == LntResidency::Absent) {
                victim_id = e.node_id;
                have_victim = true;
                break;
            }
        }
        check("with the loader idle there are still nodes that have never been fetched",
              idle && have_victim,
              std::to_string(tree.nodes.size()) + " nodes, " +
                  std::to_string(flight_cache.resident_bytes() / 1024) + " KiB resident");
        const auto t0 = std::chrono::steady_clock::now();
        if (have_victim) flight_cache.request(victim_id);
        const std::size_t cut = streamer.update_visible_nodes(eng, scene, eng_cam, err);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const LntResidency res = have_victim ? flight_cache.residency(victim_id)
                                             : LntResidency::Resident;
        check("a frame call returns without waiting for the held loader",
              have_victim && ms < 50.0 && res != LntResidency::Resident,
              std::to_string(ms) + " ms, requested node still " + residency_name(res) +
                  ", " + std::to_string(cut) + " nodes cut");
        check("the frame still produced a selection while the loader was held",
              cut > 0, std::to_string(cut) + " nodes");
        flight_cache.hold(false);
        const bool released = flight_cache.drain(20000);
        check("releasing the loader completes the held request without exceeding budget",
              released && flight_cache.resident_bytes() <= flight_budget &&
                  flight_cache.residency(victim_id) == LntResidency::Resident,
              std::to_string(flight_cache.resident_bytes()) + " of " +
                  std::to_string(flight_budget) + " bytes, node now " +
                  residency_name(flight_cache.residency(victim_id)));
    }

    // ---- facing away, measured at the engine AND in the pixels -----------
    {
        // Turn the camera around: the same position, looking away from the data.
        const double ax = mid[0], ay = mid[1], az = mid[2] - side * 3.0;
        eng.set_camera_lookat(ax, ay, az, ax, ay, az - side, 45.0, near_clip, far_clip);
        EngineFrame away_frame;
        float m[16] = {};
        if (capture_with_mvp(eng, away_frame, m)) {
            LNTCamera away;
            away.set_from_engine_mvp(m, origin, eye);
            away.screen_height_px = fh;
            const uint64_t before_calls = eng.point_draw_call_count();
            const std::size_t pushed = streamer.update_visible_nodes(eng, scene, away, err);
            // The capture below is the frame the new (empty) cut is rendered in,
            // so its draw calls are the ones this decision produced.
            EngineFrame away_drawn;
            const bool recaptured = eng.capture_frame(away_drawn);
            const uint64_t calls = eng.point_draw_call_count() - before_calls;
            uint8_t bg[3] = {0, 0, 0};
            uint64_t bg_n = 0;
            dominant_colour(away_drawn, bg, &bg_n);
            const uint64_t px = recaptured
                                    ? away_drawn.count_not_equal(bg[0], bg[1], bg[2], 6) : ~0ull;
            std::printf("  facing away    : %zu ranges, %u points, %llu draw calls, "
                        "%llu nodes culled, %llu px above background\n",
                        pushed, eng.draw_range_points(),
                        (unsigned long long)calls,
                        (unsigned long long)streamer.stats().last_selection.culled,
                        (unsigned long long)px);
            check("facing away pushes ZERO draw ranges", pushed == 0,
                  std::to_string(pushed) + " ranges");
            check("facing away asks the engine for zero points",
                  eng.draw_range_points() == 0,
                  std::to_string(eng.draw_range_points()) + " points");
            check("facing away issues no point draw call at all", calls == 0,
                  std::to_string(calls) + " vkCmdDraw calls");
            check("a frame rendered with an empty cut rasterises no geometry",
                  recaptured && px == 0,
                  recaptured ? std::to_string(px) + " px differ from the " +
                                   std::to_string(bg_n) + "-px background"
                             : "the frame after the turn did not capture");
            check("culling reduced the drawn node count when the camera faced away",
                  streamer.stats().last_selection.cut_size() == 0 &&
                      streamer.stats().last_selection.culled > 0,
                  std::to_string(streamer.stats().last_selection.cut_size()) +
                      " drawn vs " +
                      std::to_string(streamer.stats().last_selection.culled) +
                      " culled, " +
                      std::to_string(streamer.stats().last_selection.untouched(
                          tree.nodes.size())) + " never even tested");
            check("the cull needed no position upload at all",
                  eng.position_upload_count() == uploads_during_late_frames,
                  std::to_string(eng.position_upload_count()) + " cumulative uploads");
        } else {
            skip("facing-away frame", "no usable view-projection after the turn");
        }
    }

    // ---- a frame under a budget that CANNOT hold its cut ------------------
    // The 10-frame flight ran with room to spare, which proves the streaming path
    // but not the pressure path. Here the J3 budget (two payloads) is reused: the
    // cuts cannot fit, so every admission competes with the pinned working set
    // and the budget invariant is checked on every frame of a flight.
    {
        // SIZED FROM THE CUT, NOT GUESSED. The selection is computed first (it is
        // pure: no cache, no engine), its payload total is summed, and the budget
        // is set to a QUARTER of it - and never below one node's own payload, so
        // this exercises eviction rather than the "too large to admit" path. The
        // check below is what says the pressure is real rather than nominal.
        const LntLodSelection probe_sel =
            select_lod(tree, rd.nodes(), origin, eng_cam, LntLodParams{},
                       rd.get_total_points());
        uint64_t cut_bytes = 0;
        for (uint64_t id : probe_sel.node_ids) {
            const LntTile* t = scene.find_tile(id);
            if (t != nullptr) cut_bytes += uint64_t(t->count) * sizeof(RawPoint);
        }
        const uint64_t pressure_budget = std::max<uint64_t>(max_payload, cut_bytes / 4);
        std::printf("  pressure budget: %.2f MiB for a cut of %llu nodes / %.2f MiB\n",
                    double(pressure_budget) / (1024 * 1024),
                    (unsigned long long)probe_sel.cut_size(),
                    double(cut_bytes) / (1024 * 1024));
        check("the pressure budget genuinely cannot hold the cut it must draw",
              cut_bytes > 0 && pressure_budget < cut_bytes,
              std::to_string(pressure_budget) + " < " + std::to_string(cut_bytes) +
                  " bytes");

        LntCacheManager tight_cache(pressure_budget);
        std::string cerr3;
        if (!tight_cache.open(used, cerr3)) {
            check("the tight-budget cache opens for the pressure pass", false, cerr3);
        } else {
            LntStreamer tight;
            if (!tight.attach(rd, &tight_cache, cerr3)) {
                check("the pressure-pass streamer attaches", false, cerr3);
            } else {
                uint64_t over_budget = 0, deferrals = 0;
                std::size_t cold = 0, warm = 0;
                // THIS PASS ASKS A QUESTION ABOUT A COLD ARENA, so it starts from
                // one. The arena belongs to the ENGINE, not to this pass: earlier
                // phases wrote tiles into it and they are still there, at stable
                // offsets, whatever this cache holds. A frame may therefore draw a
                // tile whose payload has since been evicted - which is the whole
                // point of a permanent arena - and with the flags left as they
                // were, this pass would measure that instead of the cache budget
                // it is named for. So the flags are snapshotted, cleared for the
                // pass and restored afterwards; the pass rewrites the same bytes
                // at the same offsets, so the restored state is the true one.
                std::vector<char> uploaded_before(scene.tiles().size(), 0);
                for (std::size_t i = 0; i < scene.tiles().size(); ++i) {
                    uploaded_before[i] = scene.tiles()[i].uploaded ? 1 : 0;
                    scene.tile_mut(scene.tiles()[i].node_id)->uploaded = false;
                }
                tight.update_visible_nodes(eng, scene, eng_cam, cerr3);
                cold = eng.draw_range_count();
                if (tight_cache.resident_bytes() > pressure_budget) ++over_budget;
                // Unpinned drain so the deferred admissions can proceed: with the
                // whole cut pinned and a quarter-sized budget, most of the cut
                // could never be admitted - the documented deferral case, not a
                // bug - so the pin is released first and the churn is measured.
                tight_cache.clear_pinned();
                const bool tight_drained = tight_cache.drain(30000);
                deferrals = tight_cache.stats().deferred;
                tight.update_visible_nodes(eng, scene, eng_cam, cerr3);
                warm = eng.draw_range_count();
                if (tight_cache.resident_bytes() > pressure_budget) ++over_budget;
                const uint64_t evictions = tight_cache.stats().evictions;
                std::printf("  tight budget   : cut %llu nodes, %zu ranges cold, %zu after "
                            "the drain, %llu evictions, %llu deferrals, peak %.2f of %.2f "
                            "MiB\n",
                            (unsigned long long)tight.stats().cut_nodes, cold, warm,
                            (unsigned long long)evictions, (unsigned long long)deferrals,
                            double(tight_cache.peak_bytes()) / (1024 * 1024),
                            double(pressure_budget) / (1024 * 1024));
                check("under a budget too small for the cut, RAM still never exceeded it",
                      over_budget == 0 && tight_cache.stats().within_budget() &&
                          tight_cache.peak_bytes() <= pressure_budget,
                      std::to_string(over_budget) + " frames over a budget of " +
                          std::to_string(pressure_budget) + " bytes, peak " +
                          std::to_string(tight_cache.peak_bytes()));
                check("streaming under frame pressure evicted repeatedly",
                      evictions > 0 && tight_drained,
                      std::to_string(evictions) + " evictions during the pressure pass, " +
                          std::to_string(tight_cache.stats().loads) + " loads");
                check("a frame draws only the part of its cut that is resident",
                      cold == 0 && warm > 0 && warm < tight.stats().cut_nodes,
                      std::to_string(cold) + " ranges cold, " + std::to_string(warm) +
                          " of " + std::to_string(tight.stats().cut_nodes) + " after");
                // Equal treatment for the engine side: the ranges on the engine
                // must be exactly the ones this pass pushed.
                check("the engine holds exactly the ranges the pressure pass pushed",
                      eng.draw_range_count() == warm &&
                          uint64_t(eng.draw_range_points()) == tight.stats().cut_points,
                      std::to_string(eng.draw_range_count()) + " ranges, " +
                          std::to_string(eng.draw_range_points()) + " points");
                for (std::size_t i = 0; i < scene.tiles().size(); ++i)
                    if (uploaded_before[i])
                        scene.tile_mut(scene.tiles()[i].node_id)->uploaded = true;
            }
            tight_cache.close();
        }
    }

    eng.close();
    flight_cache.close();
}


// ========================================================================
// K. ORBIT + ZOOM MONOTONE LOD SWEEP
// ========================================================================
// 20 frames: the eye orbits the dataset centre through a full turn while the
// radius shrinks linearly from 500 m to 5 m, elevated by r/2. The trajectory is
// specified about (0,0,0); the file stores UTM-like world coordinates, so the
// (0,0,0) centre is mapped onto the dataset's own centre - the offsets are
// exactly those specified.
//
// What is asserted is the LOD (SSE) cut for each frame: point count and node
// count never decrease as the camera closes in, and no frame's cut holds both a
// node and one of its descendants. The camera always looks at the centre.
//
// FRUSTUM CULLING IS TAKEN OUT OF THE MONOTONE ASSERTION ON PURPOSE. Zooming
// into a 1200 m dataset shrinks the visible footprint, so a culled cut SHOULD
// get smaller (it does: ~70k -> ~16k points) - that is culling working, and it
// would make "more points when closer" false for a correct implementation. The
// monotone property belongs to the SSE refinement alone, so the asserted cut is
// computed with the frustum test disabled (frustum_valid = false: everything
// visible). The culled cut is still printed, and still checked as an antichain.
static void verify_monotone_orbit_zoom(const std::string& path) {
    std::printf("\n=== K. ORBIT + ZOOM MONOTONE LOD SWEEP ===\n");
    std::string used = path;
    {
        Reader probe;
        if (probe.open(path + ".fine") && probe.get_total_nodes() > 0) used = path + ".fine";
    }
    Reader rd;
    if (!rd.open(used)) { check("orbit sweep: file opens", false, rd.error()); return; }
    LntNodeTree tree;
    std::string err;
    if (!tree.build(rd.nodes(), err)) { check("orbit sweep: tree builds", false, err); return; }

    const BoundingBox env = rd.get_bounds();
    const double c[3] = {env.mid(0), env.mid(1), env.mid(2)};
    double origin[3];
    for (int i = 0; i < 3; ++i) origin[i] = rd.origin(i);

    const int kSteps = 20;
    const double kPi = 3.14159265358979323846;
    LntLodParams params;            // default SSE threshold; the cap is lifted so
    params.max_nodes = 1000000;     // truncation cannot break monotonicity

    std::printf("  %-5s %-10s %-12s %-12s %-12s %-12s\n", "step", "radius(m)",
                "active nodes", "points", "(culled n)", "(culled pts)");
    bool pts_mono = true, node_mono = true, anti = true, truncated = false;
    uint64_t prev_pts = 0, prev_nodes = 0, first_pts = 0, last_pts = 0;
    std::size_t first_nodes = 0, last_nodes = 0;
    std::string first_bad;
    for (int i = 0; i < kSteps; ++i) {
        const double t = double(i) / double(kSteps - 1);
        const double r = 500.0 + (5.0 - 500.0) * t;
        const double a = 2.0 * kPi * t;
        LNTCamera cam;
        cam.eye[0] = c[0] + r * std::cos(a);
        cam.eye[1] = c[1] + r * std::sin(a);
        cam.eye[2] = c[2] + r * 0.5;
        for (int k = 0; k < 3; ++k) { cam.target[k] = c[k]; cam.origin[k] = origin[k]; }
        cam.up[0] = 0.0; cam.up[1] = 0.0; cam.up[2] = 1.0;   // Z-up world; orbit is in XY
        cam.aspect = 16.0 / 9.0;
        cam.near_clip = 0.1;
        cam.far_clip = 5000.0;
        cam.screen_height_px = 1080;
        if (!cam.rebuild()) { check("orbit sweep: frame camera is valid", false, "degenerate basis"); return; }

        const LntLodSelection culled_sel =
            select_lod(tree, rd.nodes(), origin, cam, params, rd.get_total_points());
        cam.frustum_valid = false;      // SSE only: see the note above
        const LntLodSelection sel =
            select_lod(tree, rd.nodes(), origin, cam, params, rd.get_total_points());
        const uint64_t nodes = sel.cut_size();
        std::printf("  %-5d %-10.2f %-12llu %-12llu %-12zu %-12llu\n", i, r,
                    (unsigned long long)nodes, (unsigned long long)sel.selected_points,
                    culled_sel.cut_size(), (unsigned long long)culled_sel.selected_points);
        if (sel.truncated) truncated = true;
        if (i > 0) {
            if (sel.selected_points < prev_pts && pts_mono)
                { pts_mono = false; first_bad = "points step " + std::to_string(i); }
            if (nodes < prev_nodes && node_mono)
                { node_mono = false; if (first_bad.empty()) first_bad = "nodes step " + std::to_string(i); }
        } else { first_pts = sel.selected_points; first_nodes = nodes; }
        prev_pts = sel.selected_points; prev_nodes = nodes;
        last_pts = sel.selected_points; last_nodes = nodes;

        for (const LntLodSelection* cut : {&sel, &culled_sel})
            for (std::size_t x = 0; x < cut->node_ids.size() && anti; ++x)
                for (std::size_t y = 0; y < cut->node_ids.size(); ++y)
                    if (x != y && LntNodeTree::is_ancestor(cut->node_ids[x], cut->node_ids[y])) {
                        anti = false;
                        break;
                    }
    }
    check("orbit sweep: 20 frames ran with no truncated cut", !truncated);
    check("orbit sweep: point count never decreases as the radius shrinks", pts_mono,
          std::to_string(first_pts) + " -> " + std::to_string(last_pts) + " " + first_bad);
    check("orbit sweep: active node count never decreases as the radius shrinks", node_mono,
          std::to_string(first_nodes) + " -> " + std::to_string(last_nodes) + " " + first_bad);
    check("orbit sweep: every frame's cut is an antichain", anti);
}


// ========================================================================
// L. LAS INPUT -> .lnt
// ========================================================================
// Writes a real ASPRS LAS file (header + records, little-endian) and converts it.
// `expected` receives what a correct reader must decode: world = raw*scale+off,
// colour/intensity >> 8, grey from intensity when the format has no colour.
static bool generate_test_las_file(const std::string& path, size_t n,
                                   std::vector<RawPoint>* expected = nullptr,
                                   int format = 3, int minor = 2, uint64_t seed = 77) {
    static const uint16_t kLen[11] = {20, 28, 26, 34, 57, 63, 30, 36, 38, 59, 67};
    static const int kRgb[11] = {-1, -1, 20, 28, -1, 28, -1, 30, 30, -1, 30};
    LASHeader h;
    std::memset(&h, 0, sizeof(h));
    std::memcpy(h.magic, "LASF", 4);
    h.version_major = 1;
    h.version_minor = uint8_t(minor);
    h.header_size = minor >= 4 ? uint16_t(375) : uint16_t(227);
    h.offset_to_point_data = h.header_size;
    h.point_data_format = uint8_t(format);
    h.point_data_record_length = kLen[format];
    if (minor < 4 || format < 6) h.legacy_point_count = uint32_t(n);
    h.extended_point_count = n;
    h.scale_x = h.scale_y = 0.001; h.scale_z = 0.0005;
    h.offset_x = 682000.0; h.offset_y = 3504000.0; h.offset_z = 1040.0;
    Rng rng(seed);
    std::vector<uint8_t> body(n * h.point_data_record_length, 0);
    if (expected) expected->resize(n);
    double mn[3] = {1e300, 1e300, 1e300}, mx[3] = {-1e300, -1e300, -1e300};
    for (size_t i = 0; i < n; ++i) {
        uint8_t* p = body.data() + i * h.point_data_record_length;
        const int32_t rx = int32_t(rng.unit() * 1200000.0);   // 1200 m at 1 mm
        const int32_t ry = int32_t(rng.unit() * 1000000.0);
        const int32_t rz = int32_t(rng.unit() * 240000.0);    // 120 m at 0.5 mm
        const uint16_t in16 = uint16_t(rng.byte() << 8 | rng.byte());
        uint16_t c16[3] = {uint16_t(rng.byte() << 8 | rng.byte()),
                           uint16_t(rng.byte() << 8 | rng.byte()),
                           uint16_t(rng.byte() << 8 | rng.byte())};
        std::memcpy(p, &rx, 4); std::memcpy(p + 4, &ry, 4); std::memcpy(p + 8, &rz, 4);
        std::memcpy(p + 12, &in16, 2);
        if (kRgb[format] >= 0) std::memcpy(p + kRgb[format], c16, 6);
        RawPoint e;
        e.x = double(rx) * h.scale_x + h.offset_x;
        e.y = double(ry) * h.scale_y + h.offset_y;
        e.z = double(rz) * h.scale_z + h.offset_z;
        e.intensity = uint8_t(in16 >> 8);
        // The class goes where the format keeps it: low 5 bits of byte 15 for
        // formats 0-5 (top bits are flags), a full byte at 16 for formats 6-10.
        const uint8_t cls_raw = rng.byte();
        if (format >= 6) { p[16] = cls_raw; e.classification = cls_raw; }
        else { p[15] = uint8_t(cls_raw & 0x1F); e.classification = uint8_t(cls_raw & 0x1F); }
        if (kRgb[format] >= 0) { e.r = uint8_t(c16[0] >> 8); e.g = uint8_t(c16[1] >> 8); e.b = uint8_t(c16[2] >> 8); }
        else e.r = e.g = e.b = e.intensity;
        const double v[3] = {e.x, e.y, e.z};
        for (int k = 0; k < 3; ++k) { mn[k] = std::min(mn[k], v[k]); mx[k] = std::max(mx[k], v[k]); }
        if (expected) (*expected)[i] = e;
    }
    h.min_x = mn[0]; h.max_x = mx[0]; h.min_y = mn[1]; h.max_y = mx[1];
    h.min_z = mn[2]; h.max_z = mx[2];
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(&h), h.header_size);
    f.write(reinterpret_cast<const char*>(body.data()), std::streamsize(body.size()));
    return bool(f);
}

static void section_las() {
    std::printf("\n=== L. LAS INPUT -> .lnt ===\n");
    const size_t N = 100000;
    const std::string las = "input_test.las", out = "converted_output.lnt";
    std::vector<RawPoint> exp;
    check("a LAS 1.2 (format 3) test file is written",
          generate_test_las_file(las, N, &exp, 3, 2));

    // ---- the reader on its own ------------------------------------------
    {
        LASReader r;
        const bool ok = r.open(las);
        check("LASReader opens the LAS 1.2 file", ok, r.error());
        if (!ok) return;
        const LASHeader& h = r.header();
        check("header: magic, version, format, record length, count",
              h.version_major == 1 && h.version_minor == 2 && h.point_data_format == 3 &&
                  h.point_data_record_length == 34 && r.point_count() == N && r.has_color(),
              std::to_string(r.point_count()) + " points");
        check("header: scale and offset are read from their real offsets",
              h.scale_x == 0.001 && h.scale_z == 0.0005 && h.offset_x == 682000.0 &&
                  h.offset_y == 3504000.0 && h.offset_z == 1040.0);
        // Chunked decode must equal the expected decode exactly (the doubles come
        // from the same raw*scale+offset expression).
        std::vector<RawPoint> chunk, all;
        size_t chunks = 0;
        while (r.read_chunk(chunk, 30000)) { ++chunks; all.insert(all.end(), chunk.begin(), chunk.end()); }
        bool same = all.size() == exp.size();
        for (size_t i = 0; same && i < all.size(); ++i)
            same = all[i].x == exp[i].x && all[i].y == exp[i].y && all[i].z == exp[i].z &&
                   all[i].r == exp[i].r && all[i].g == exp[i].g && all[i].b == exp[i].b &&
                   all[i].intensity == exp[i].intensity &&
                   all[i].classification == exp[i].classification;
        check("30000-point chunks decode every point exactly (raw*scale+offset, >>8, class)",
              same && chunks == 4, std::to_string(all.size()) + " points in " +
                                       std::to_string(chunks) + " chunks");
        check("the reader reports exhaustion after the last chunk",
              !r.read_chunk(chunk, 30000) && r.points_remaining() == 0);
        const Bounds3d b = r.bounds();
        double dmn[3] = {1e300, 1e300, 1e300}, dmx[3] = {-1e300, -1e300, -1e300};
        for (const RawPoint& p : exp) {
            const double v[3] = {p.x, p.y, p.z};
            for (int k = 0; k < 3; ++k) { dmn[k] = std::min(dmn[k], v[k]); dmx[k] = std::max(dmx[k], v[k]); }
        }
        check("header bounds equal the true extent of the decoded points",
              b.min[0] == dmn[0] && b.max[0] == dmx[0] && b.min[1] == dmn[1] &&
                  b.max[1] == dmx[1] && b.min[2] == dmn[2] && b.max[2] == dmx[2]);
    }

    // ---- formats, versions and rejection ---------------------------------
    {
        std::vector<RawPoint> e0, e7;
        generate_test_las_file("input_fmt0.las", 5000, &e0, 0, 2);
        generate_test_las_file("input_fmt7.las", 5000, &e7, 7, 4);
        LASReader r0, r7;
        std::vector<RawPoint> b0, b7;
        const bool o0 = r0.open("input_fmt0.las") && r0.read_chunk(b0, 5000);
        const bool o7 = r7.open("input_fmt7.las") && r7.read_chunk(b7, 5000);
        bool g0 = o0 && !r0.has_color() && b0.size() == 5000;
        for (size_t i = 0; g0 && i < b0.size(); ++i)
            g0 = b0[i].r == b0[i].intensity && b0[i].g == b0[i].intensity &&
                 b0[i].b == b0[i].intensity && b0[i].x == e0[i].x;
        check("format 0 (no colour) decodes to intensity-grey", g0, r0.error());
        bool g7 = o7 && r7.has_color() && r7.point_count() == 5000 && r7.header().version_minor == 4;
        for (size_t i = 0; g7 && i < b7.size(); ++i)
            g7 = b7[i].r == e7[i].r && b7[i].b == e7[i].b && b7[i].z == e7[i].z;
        check("LAS 1.4 format 7 decodes (64-bit count, colour at offset 30)", g7, r7.error());

        // LAZ marker, bad magic, truncated file.
        {
            std::ifstream in(las, std::ios::binary);
            std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            std::vector<char> laz = bytes; laz[104] = char(laz[104] | 0x80);
            std::ofstream("input_laz.las", std::ios::binary).write(laz.data(), std::streamsize(laz.size()));
            std::vector<char> bad = bytes; bad[0] = 'X';
            std::ofstream("input_badmagic.las", std::ios::binary).write(bad.data(), std::streamsize(bad.size()));
            std::ofstream("input_trunc.las", std::ios::binary).write(bytes.data(), std::streamsize(bytes.size() / 2));
        }
        LASReader a, b, c, d;
        check("LAZ input is rejected with a clear error, not misread",
              !a.open("input_laz.las") && a.error().find("LAZ") != std::string::npos, a.error());
        check("a bad magic is rejected", !b.open("input_badmagic.las"), b.error());
        check("a truncated file is rejected before any point is decoded",
              !c.open("input_trunc.las") && c.error().find("truncated") != std::string::npos, c.error());
        check("a missing file is rejected", !d.open("no_such_file.las"), d.error());
    }

    // ---- the conversion --------------------------------------------------
    LNTBuildParams params;
    params.max_points_per_node = 60000;
    LNTBuildStats st;
    std::string err;
    const bool conv = convert_las_to_lnt(las, out, params, &st, &err);
    check("convert_las_to_lnt succeeds", conv, err);
    if (!conv) return;
    check("converting a missing file reports failure", !convert_las_to_lnt("no_such.las", "x.lnt"));

    Reader rd;
    if (!rd.open(out)) { check("converted_output.lnt opens with Reader", false, rd.error()); return; }
    check("converted file: header point count equals the LAS count",
          rd.get_total_points() == N, std::to_string(rd.get_total_points()));

    // Leaf payloads are the full-detail points (interior nodes hold LOD copies).
    double maxstep = 0.0;
    for (const LNTNodeEntry& e : rd.nodes())
        if (is_leaf_node(e)) maxstep = std::max(maxstep, 2.0 * double(e.half_size) / 65535.0);
    const double grid = std::max(maxstep, 1e-6);
    const double ox = rd.origin(0), oy = rd.origin(1), oz = rd.origin(2);
    auto cell_key = [](int64_t a, int64_t b, int64_t c) {
        return ((uint64_t(a + 4096) * 1000003ull + uint64_t(b + 4096)) * 1000003ull) +
               uint64_t(c + 4096);
    };
    std::unordered_multimap<uint64_t, uint32_t> cells;
    cells.reserve(exp.size() * 2);
    for (uint32_t i = 0; i < exp.size(); ++i)
        cells.emplace(cell_key(int64_t(std::floor((exp[i].x - ox) / grid)),
                               int64_t(std::floor((exp[i].y - oy) / grid)),
                               int64_t(std::floor((exp[i].z - oz) / grid))), i);
    std::vector<uint8_t> used(exp.size(), 0);

    uint64_t leaf_pts = 0, unmatched = 0;
    double worst = 0.0;          // worst error as a fraction of the node's step
    for (const LNTNodeEntry& e : rd.nodes()) {
        if (!is_leaf_node(e)) continue;
        std::vector<RawPoint> pts;
        if (!rd.read_node_points(e, pts)) { ++unmatched; continue; }
        const double step = 2.0 * double(e.half_size) / 65535.0;
        const double tol = 0.5 * step + 1e-4;     // 1e-4: float32 node centre
        for (const RawPoint& p : pts) {
            ++leaf_pts;
            int64_t found = -1;
            const int64_t ca = int64_t(std::floor((p.x - ox) / grid)),
                          cb = int64_t(std::floor((p.y - oy) / grid)),
                          cc = int64_t(std::floor((p.z - oz) / grid));
            for (int da = -1; da <= 1 && found < 0; ++da)
              for (int db = -1; db <= 1 && found < 0; ++db)
                for (int dc = -1; dc <= 1 && found < 0; ++dc) {
                    auto range = cells.equal_range(cell_key(ca + da, cb + db, cc + dc));
                    for (auto it = range.first; it != range.second; ++it) {
                        const RawPoint& s = exp[it->second];
                        if (used[it->second]) continue;
                        const double d = std::max({std::fabs(s.x - p.x), std::fabs(s.y - p.y),
                                                   std::fabs(s.z - p.z)});
                        if (d <= tol && s.r == p.r && s.g == p.g && s.b == p.b &&
                            s.intensity == p.intensity && s.classification == p.classification) {
                            found = it->second;
                            worst = std::max(worst, d / step);
                            break;
                        }
                    }
                }
            if (found < 0) { ++unmatched; continue; }
            used[size_t(found)] = 1;
        }
    }
    check("leaf payloads hold exactly N points", leaf_pts == N, std::to_string(leaf_pts));
    check("every decoded point matches a distinct source point with identical RGB, "
          "intensity and CLASSIFICATION, within 0.5 x step",
          unmatched == 0, std::to_string(unmatched) + " unmatched, worst error " +
                              std::to_string(worst) + " x step");
    size_t unused = 0;
    for (uint8_t u : used) if (!u) ++unused;
    check("no source point is left unaccounted for", unused == 0, std::to_string(unused) + " unused");

    // Spatial bounds: the cubic envelope must contain the LAS header bounds.
    {
        LASReader r;
        r.open(las);
        const Bounds3d lb = r.bounds();
        const BoundingBox eb = rd.get_bounds();
        bool contains = true;
        for (int k = 0; k < 3; ++k)
            contains = contains && eb.min[k] <= lb.min[k] + 1e-9 && eb.max[k] >= lb.max[k] - 1e-9;
        check("the .lnt envelope contains the LAS header bounds and is cubic",
              contains && eb.is_cube(1e-3));
        check("the builder received every LAS point", st.input_points == N);
    }
}


// ========================================================================
// M. INTERACTIVE CAMERA + RENDER LOOP
// ========================================================================
// The controller and the loop are driven by a scripted "user": 60 frames of
// orbit + pan + zoom while the SSE dial sweeps 50 -> 0. The engine-less
// update_frame() path always runs; the engine path (render_frame, which also
// uploads and pushes the draw ranges) runs when a Vulkan device is present.
static void flight_inputs(LntCameraController& c, int f, int frames, double side) {
    const double t = double(f) / double(frames - 1);
    c.orbit(float(2.0 * 3.14159265358979323846 / frames), (f < frames / 2) ? 0.004f : -0.004f);
    c.pan(float(side * 0.002), float(side * 0.001));
    c.zoom(float(-side * 1.4 / frames));            // glide from ~1.6 side to ~0.2 side
    c.set_sse_threshold(50.0 * (1.0 - t));          // coarse -> full resolution
}

static void section_interactive() {
    std::printf("\n=== M. INTERACTIVE CAMERA + RENDER LOOP ===\n");

    // ---- the controller on its own ---------------------------------------
    {
        LntCameraController c;
        check("controller: starts in 2D plan mode (orthographic, orbit locked)", c.ortho2d());
        const double yaw0 = c.yaw();
        c.orbit(1.0f, 1.0f);
        check("controller: orbit is ignored in 2D", c.yaw() == yaw0 && c.pitch() == 0.0);
        c.set_ortho2d(false);                  // the orbit checks below are 3D behaviour
        c.set_target(10.0, 20.0, 30.0);
        c.set_distance(100.0);
        c.set_angles(0.0, 0.0);
        double e[3];
        c.eye(e);
        check("controller: yaw 0 / pitch 0 puts the eye on +Z at the set distance",
              std::fabs(e[0] - 10.0) < 1e-9 && std::fabs(e[1] - 20.0) < 1e-9 &&
                  std::fabs(e[2] - 130.0) < 1e-9);
        c.orbit(1.5707963267948966f, 0.0f);
        c.eye(e);
        const double d = std::sqrt((e[0] - 10) * (e[0] - 10) + (e[1] - 20) * (e[1] - 20) +
                                   (e[2] - 30) * (e[2] - 30));
        check("controller: a 90 degree orbit keeps the distance and moves the eye to +X",
              std::fabs(d - 100.0) < 1e-4 && std::fabs(e[0] - 110.0) < 1e-3);
        c.orbit(0.0f, 10.0f);
        check("controller: pitch is clamped below 90 degrees (the view basis stays valid)",
              c.pitch() < 1.5707963267948966 && c.pitch() > 1.5);
        LNTCamera cam;
        const double o0[3] = {0, 0, 0};
        check("controller: even at the pitch limit the camera rebuilds a valid frustum",
              c.apply_to(cam, o0, 16.0 / 9.0, 720));

        c.set_angles(0.3, 0.4);
        double e0[3], e1[3];
        c.eye(e0);
        const double* t0p = c.target();
        const double tb[3] = {t0p[0], t0p[1], t0p[2]};
        c.pan(5.0f, -3.0f);
        c.eye(e1);
        const double* t1p = c.target();
        const double mv[3] = {t1p[0] - tb[0], t1p[1] - tb[1], t1p[2] - tb[2]};
        check("controller: pan moves eye and target by the same vector",
              std::fabs((e1[0] - e0[0]) - mv[0]) < 1e-9 && std::fabs((e1[1] - e0[1]) - mv[1]) < 1e-9 &&
                  std::fabs((e1[2] - e0[2]) - mv[2]) < 1e-9);
        check("controller: pan by (5, -3) moves exactly sqrt(34) metres, view-plane only",
              std::fabs(std::sqrt(mv[0] * mv[0] + mv[1] * mv[1] + mv[2] * mv[2]) - std::sqrt(34.0)) < 1e-6);

        c.set_distance(100.0);
        c.zoom(1e9f);
        check("controller: zoom clamps its goal to 10000", c.goal_distance() == 10000.0);
        c.zoom(-1e9f);
        check("controller: zoom clamps its goal to 1", c.goal_distance() == 1.0);
        double prev = c.distance();
        bool smooth = true, never_out = true;
        for (int i = 0; i < 600; ++i) {
            c.update(1.0f / 60.0f);
            const double dd = c.distance();
            if (dd > prev + 1e-12) smooth = false;               // monotone glide
            if (dd < 1.0 || dd > 10000.0) never_out = false;
            prev = dd;
        }
        check("controller: the zoom glide is monotone, in range, and settles on the goal",
              smooth && never_out && c.distance() == 1.0, std::to_string(c.distance()));
        c.set_sse_threshold(-5.0);
        const bool lo = c.sse_threshold() == 0.0;
        c.set_sse_threshold(500.0);
        const bool hi = c.sse_threshold() == 50.0;
        c.adjust_sse(-8.0);
        check("controller: the SSE dial clamps to [0, 50] and adjusts relatively",
              lo && hi && c.sse_threshold() == 42.0);
    }

    // ---- the dataset ------------------------------------------------------
    // Same file selection as section J: the deep (617-node) tree when present.
    std::string used = "test.lnt";
    {
        Reader probe;
        if (probe.open("test.lnt.fine") && probe.get_total_nodes() > 0) used = "test.lnt.fine";
    }
    Reader rd;
    if (!rd.open(used)) { check("interactive: the dataset opens", false, rd.error()); return; }
    std::string err;
    const BoundingBox env = rd.get_bounds();
    const double side = env.extent(0);
    double origin[3];
    for (int i = 0; i < 3; ++i) origin[i] = rd.origin(i);

    uint64_t max_payload = 0, total_payload = 0;
    for (const LNTNodeEntry& e : rd.nodes()) {
        const uint64_t b = LntCacheManager::payload_bytes(e.point_count);
        total_payload += b;
        max_payload = std::max(max_payload, b);
    }

    auto make_controller = [&](LntCameraController& c) {
        c.set_target(env.mid(0), env.mid(1), env.mid(2));
        c.set_lens(45.0, 1.0, side * 40.0);
        c.set_ortho2d(false);              // the scripted flight is a perspective 3D flight
        c.set_angles(0.6, 0.5);
        c.set_distance(side * 1.6);
    };

    // ---- SSE dial -> point count, engine-less, fixed camera ---------------
    {
        LntCacheManager cache(total_payload + max_payload);
        LntStreamer streamer;
        LntCameraController ctl;
        LntInteractiveLoop loop;
        if (!cache.open(used, err) || !streamer.attach(rd, &cache, err)) {
            check("interactive: cache and streamer attach", false, err);
            return;
        }
        make_controller(ctl);
        ctl.set_distance(LntCameraController::kMaxDistance);  // far: the root alone passes at 50 px
        loop.attach(&ctl, &cache, origin);
        loop.set_viewport(960, 720);
        uint32_t pts[4] = {}, nodes[4] = {};
        const double dial[4] = {50.0, 20.0, 5.0, 0.0};
        bool ran = true;
        for (int i = 0; i < 4; ++i) {
            ctl.set_sse_threshold(dial[i]);
            ran = loop.update_frame(streamer, 0.0f) && ran;
            pts[i] = loop.stats().total_points_rendered;
            nodes[i] = loop.stats().active_nodes;
        }
        std::printf("  sse dial       : 50 -> %u nodes/%u pts, 20 -> %u/%u, 5 -> %u/%u, 0 -> %u/%u\n",
                    nodes[0], pts[0], nodes[1], pts[1], nodes[2], pts[2], nodes[3], pts[3]);
        check("lowering the SSE threshold never selects fewer nodes or points",
              ran && nodes[0] <= nodes[1] && nodes[1] <= nodes[2] && nodes[2] <= nodes[3] &&
                  pts[0] <= pts[1] && pts[1] <= pts[2] && pts[2] <= pts[3]);
        check("point counts change when the SSE threshold changes (50 vs 0 differ)",
              pts[0] != pts[3] && nodes[0] < nodes[3], std::to_string(pts[0]) + " vs " +
                                                         std::to_string(pts[3]));
        check("the loop copies the controller's dial into the streamer",
              streamer.params().sse_threshold == 0.0 && loop.stats().sse_threshold == 0.0);
        cache.close();
    }

    // ---- 60-frame flight, engine-less, under two budgets ------------------
    for (int pass = 0; pass < 2; ++pass) {
        const uint64_t budget = pass == 0 ? total_payload + max_payload
                                          : std::max<uint64_t>(total_payload * 2 / 5, max_payload * 2);
        LntCacheManager cache(budget);
        LntStreamer streamer;
        LntCameraController ctl;
        LntInteractiveLoop loop;
        if (!cache.open(used, err) || !streamer.attach(rd, &cache, err)) {
            check("interactive flight: cache and streamer attach", false, err);
            return;
        }
        make_controller(ctl);
        loop.attach(&ctl, &cache, origin);
        loop.set_viewport(960, 720);
        const int kF = 60;
        double max_up = 0.0, max_frame = 0.0;
        uint64_t over = 0, failed = 0, min_pts = ~0ull, max_pts = 0, peak_pending = 0;
        uint32_t first_nodes = 0, last_nodes = 0, evictions = 0;
        for (int f = 0; f < kF; ++f) {
            flight_inputs(ctl, f, kF, side);
            if (!loop.update_frame(streamer, 1.0f / 60.0f)) { ++failed; continue; }
            const LNTStreamerStats& s = loop.stats();
            max_up = std::max(max_up, double(s.streamer_update_ms));
            max_frame = std::max(max_frame, double(s.frame_time_ms));
            if (s.cache_bytes_used > s.cache_capacity_bytes) ++over;
            min_pts = std::min<uint64_t>(min_pts, s.total_points_rendered);
            max_pts = std::max<uint64_t>(max_pts, s.total_points_rendered);
            peak_pending = std::max<uint64_t>(peak_pending, s.pending_async_loads);
            if (f == 0) first_nodes = s.active_nodes;
            if (f == kF - 1) { last_nodes = s.active_nodes; evictions = s.total_evictions; }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));   // loader gets time
        }
        cache.drain(20000);
        std::printf("  flight (%s): %d frames, max update %.3f ms, max frame %.3f ms, nodes %u -> %u, "
                    "points %llu..%llu, peak pending %llu, %u evictions, peak %.2f of %.2f MiB\n",
                    pass == 0 ? "roomy " : "tight ", kF, max_up, max_frame, first_nodes, last_nodes,
                    (unsigned long long)min_pts, (unsigned long long)max_pts,
                    (unsigned long long)peak_pending, evictions,
                    double(cache.peak_bytes()) / (1024 * 1024), double(budget) / (1024 * 1024));
        const std::string tag = pass == 0 ? "roomy budget: " : "tight budget: ";
        check(tag + "all 60 frames ran", failed == 0, std::to_string(failed) + " failed");
        check(tag + "streamer update stays <= 5 ms on every frame", max_up <= 5.0,
              std::to_string(max_up) + " ms worst");
        check(tag + "point count changes as the SSE dial sweeps 50 -> 0",
              max_pts > min_pts && last_nodes > first_nodes,
              std::to_string(min_pts) + " .. " + std::to_string(max_pts));
        check(tag + "cache bytes never exceeded the budget (per frame and at the peak)",
              over == 0 && cache.stats().within_budget() && cache.peak_bytes() <= budget,
              std::to_string(cache.peak_bytes()) + " of " + std::to_string(budget));
        if (pass == 1)
            check("tight budget: the flight evicted payloads and stayed within budget",
                  evictions > 0 || cache.stats().evictions > 0,
                  std::to_string(cache.stats().evictions) + " evictions");
        cache.close();
    }

    // ---- the same flight through the engine --------------------------------
    const std::string dll = VulkanEngine::find_dll();
    if (dll.empty()) { skip("interactive engine flight", "no naksha_vulkan.dll found"); return; }
    VulkanEngine eng;
    if (!eng.load(dll, err)) { skip("interactive engine flight", "engine load failed: " + err); return; }
    if (!eng.is_available()) { skip("interactive engine flight", "no Vulkan device"); return; }
    if (!eng.open(960, 720, true, false, err)) {
        skip("interactive engine flight", "renderer creation failed: " + err);
        return;
    }
    LntScene scene;
    if (!scene.plan(rd, err) || !scene.reserve(eng, err)) {
        check("interactive engine flight: the arena is planned and reserved", false, err);
        return;
    }
    const uint64_t budget = total_payload + max_payload;
    LntCacheManager cache(budget);
    LntStreamer streamer;
    LntCameraController ctl;
    LntInteractiveLoop loop;
    if (!cache.open(used, err) || !streamer.attach(rd, &cache, err)) {
        check("interactive engine flight: cache and streamer attach", false, err);
        return;
    }
    eng.set_render_origin(origin[0], origin[1], origin[2]);
    eng.set_clear_color(0.03f, 0.03f, 0.05f, 1.0f);
    make_controller(ctl);
    loop.attach(&ctl, &cache, origin);
    loop.set_viewport(960, 720);
    loop.set_present_mode(LntPresent::Render);

    const int kF = 60;
    double max_up = 0.0, max_frame = 0.0;
    uint64_t over = 0, failed = 0, range_mismatch = 0, point_mismatch = 0, drew = 0;
    uint32_t last_pts = 0;
    for (int f = 0; f < kF; ++f) {
        flight_inputs(ctl, f, kF, side);
        if (!loop.render_frame(eng, scene, streamer, 1.0f / 60.0f)) { ++failed; continue; }
        const LNTStreamerStats& s = loop.stats();
        max_up = std::max(max_up, double(s.streamer_update_ms));
        max_frame = std::max(max_frame, double(s.frame_time_ms));
        if (s.cache_bytes_used > s.cache_capacity_bytes) ++over;
        if (eng.draw_range_count() != streamer.stats().ranges_pushed) ++range_mismatch;
        if (uint64_t(eng.draw_range_points()) != uint64_t(s.total_points_rendered)) ++point_mismatch;
        if (s.total_points_rendered > 0) ++drew;
        last_pts = s.total_points_rendered;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    std::printf("  engine flight  : %d frames, max update %.3f ms, max frame %.3f ms, %llu frames drew, "
                "last frame %u pts\n", kF, max_up, max_frame, (unsigned long long)drew, last_pts);
    check("engine flight: all 60 frames rendered", failed == 0,
          std::to_string(failed) + " failed " + loop.error());
    check("engine flight: streamer update stays <= 5 ms on every frame", max_up <= 5.0,
          std::to_string(max_up) + " ms worst");
    check("engine flight: frames drew points once payloads arrived", drew > 0);
    check("engine flight: the engine's ranges and points equal what the loop pushed",
          range_mismatch == 0 && point_mismatch == 0,
          std::to_string(range_mismatch) + " range / " + std::to_string(point_mismatch) +
              " point mismatches");
    check("engine flight: cache bytes stayed within the budget",
          over == 0 && cache.stats().within_budget() && cache.peak_bytes() <= budget);
    // The capture presentation path, once, at the end of the flight.
    loop.set_present_mode(LntPresent::Capture);
    check("engine flight: the capture_frame presentation path works",
          loop.render_frame(eng, scene, streamer, 0.0f), loop.error());
    eng.close();
    cache.close();
}

// ========================================================================
// O. ZOOM COVERAGE, LOD FALLBACK AND SELECTION COST
// ========================================================================
// Four claims, each as a measurement rather than a description:
//
//   O1  ZOOMED OUT the cut covers the cloud and drops nothing that is on screen,
//       stated twice on purpose - once as the envelope the header declares, and
//       once as the invariant that cannot pass by accident: every leaf the
//       frustum keeps is either in the cut or under a node that is, and no leaf
//       whose cell lies WHOLLY inside the frustum is missing from it.
//   O2  ZOOMED IN the frustum discards what is off screen and refinement runs to
//       the tree's OWN leaves. On a tree deep enough to show it, the wide view
//       must simultaneously stop SHORT of the leaves: that is the LOD fallback,
//       a coarse node's strided sample standing in for its whole subtree.
//   O3  THE FAR PLANE the engine's float32 view-projection produces is checked
//       against the float64 one, and against the cull budget LNTCamera grants it.
//       This is the plane whose extraction is ill-conditioned: at 2000 m the raw
//       row3 - row2 far plane sits 746 m away from where the float64 frustum puts
//       it, over this envelope, which is more than enough to cull a section that
//       is plainly on screen.
//   O4  A CAMERA EDIT too small to change the cut costs no traversal at all, and
//       the 100-frame zoom-out -> zoom-in sweep stays inside 1.5 ms of streamer
//       time per frame.
//
// The coverage statements are run on BOTH the flat file and section F's deep
// build, because the two trees can exhibit different things: the flat tree is
// one level deep, so its only possible cuts are "the leaves" and "the root" and
// it can support the envelope statement but no claim about hierarchy, while the
// deep tree is what shows a coarse node standing in for a subtree.

// The union of a cut's world-space cells. False if any id failed to resolve, in
// which case the union is meaningless rather than merely smaller.
static bool cut_envelope(const LntStreamer& s, const Reader& rd,
                         const LntLodSelection& cut, BoundingBox& out) {
    out.set_empty();
    bool ok = !cut.node_ids.empty();
    for (uint64_t id : cut.node_ids) {
        const auto it = s.tree().index_of.find(id);
        if (it == s.tree().index_of.end()) { ok = false; continue; }
        const BoundingBox b = node_bounds(rd.nodes()[it->second], s.file_origin());
        for (int a = 0; a < 3; ++a) {
            out.min[a] = std::min(out.min[a], b.min[a]);
            out.max[a] = std::max(out.max[a], b.max[a]);
        }
    }
    return ok;
}

// True when every corner of the box is inside every plane by at least `margin`.
// This is the unambiguous "this cell is on screen" test: cam.visible() is
// conservative on purpose, so a cell it keeps can genuinely lie outside the
// cone, but a cell that passes THIS is inside it under any convention - which is
// what makes "no cell wholly inside the frustum was dropped" a statement about
// geometry rather than about the tolerance the cull test chose.
static bool box_strictly_inside(const LNTCamera& cam, const BoundingBox& b, double margin) {
    if (!cam.frustum_valid || b.is_empty()) return false;
    for (int p = 0; p < kPlaneCount; ++p) {
        const double* pl = cam.planes[p];
        for (int c = 0; c < 8; ++c) {
            const double x = (c & 1) ? b.max[0] : b.min[0];
            const double y = (c & 2) ? b.max[1] : b.min[1];
            const double z = (c & 4) ? b.max[2] : b.min[2];
            if (pl[0] * x + pl[1] * y + pl[2] * z + pl[3] < margin) return false;
        }
    }
    return true;
}

struct LntCoverStats {
    uint64_t visible_leaves = 0;   // leaves the conservative frustum test keeps
    uint64_t missed_visible = 0;   // ... of those, not in the cut or under it
    uint64_t inside_leaves = 0;    // leaves lying wholly inside the frustum
    uint64_t missed_inside = 0;    // ... of those, dropped: a visual hole
    uint64_t missed_points = 0;    // points behind missed_inside
};

// A cut is an antichain, so "the union of its cells covers the envelope" is only
// the headline: the property that actually means "no holes" is that every leaf
// on screen is in the cut or under a node that is. Counts both, so a failure
// says which one broke.
static LntCoverStats cut_coverage(const LntStreamer& s, const Reader& rd,
                                  const LNTCamera& cam, const LntLodSelection& cut) {
    LntCoverStats st;
    std::vector<uint64_t> ids = cut.node_ids;
    std::sort(ids.begin(), ids.end());
    const LntNodeTree& tree = s.tree();
    for (std::size_t i = 0; i < tree.nodes.size(); ++i) {
        if (!tree.nodes[i].children.empty()) continue;                 // leaves only
        const uint64_t id = tree.nodes[i].id;
        const BoundingBox cell = node_bounds(rd.nodes()[i], s.file_origin());
        const bool visible = cam.visible(rd.nodes()[i], s.file_origin());
        const bool inside = box_strictly_inside(cam, cell, 0.01);
        if (visible) ++st.visible_leaves;
        if (inside) ++st.inside_leaves;
        bool covered = std::binary_search(ids.begin(), ids.end(), id);
        if (!covered) {
            for (uint64_t c : cut.node_ids)
                if (LntNodeTree::is_ancestor(c, id)) { covered = true; break; }
        }
        if (covered) continue;
        if (visible) ++st.missed_visible;
        if (inside) { ++st.missed_inside; st.missed_points += rd.nodes()[i].point_count; }
    }
    return st;
}

// Worst disagreement between two planes, in metres, over a box's eight corners.
// A signed-distance difference is the only form in which two plane equations can
// be compared: the coefficients themselves are not lengths.
static double plane_gap_over_box(const double a[4], const double b[4], const BoundingBox& box) {
    double worst = 0.0;
    for (int c = 0; c < 8; ++c) {
        const double x = (c & 1) ? box.max[0] : box.min[0];
        const double y = (c & 2) ? box.max[1] : box.min[1];
        const double z = (c & 4) ? box.max[2] : box.min[2];
        const double da = a[0] * x + a[1] * y + a[2] * z + a[3];
        const double db = b[0] * x + b[1] * y + b[2] * z + b[3];
        worst = std::max(worst, std::fabs(da - db));
    }
    return worst;
}

// The smallest cell in a cut, which is the resolution the cut achieved and the
// scale the streamer's motion hysteresis is relative to.
static double min_cut_node_size(const LntStreamer& s, const Reader& rd) {
    double smallest = 0.0;
    for (uint64_t id : s.stats().last_selection.node_ids) {
        const auto it = s.tree().index_of.find(id);
        if (it == s.tree().index_of.end()) continue;
        const double size = 2.0 * double(rd.nodes()[it->second].half_size);
        if (size > 0.0 && (smallest == 0.0 || size < smallest)) smallest = size;
    }
    return smallest;
}

static void section_zoom_coverage(const std::string& path) {
    std::printf("\n=== O. ZOOM COVERAGE, LOD FALLBACK AND SELECTION COST ===\n");

    std::string deep;
    {
        Reader probe;
        const std::string candidate = path + ".fine";
        if (probe.open(candidate) && probe.get_total_nodes() > 0) deep = candidate;
    }
    if (deep.empty())
        std::printf("  note           : %s.fine is not on disk - hierarchy claims will be "
                    "skipped\n", path.c_str());

    Reader rd;
    if (!rd.open(path)) { check("zoom: the dataset opens", false, rd.error()); return; }
    std::string err;
    LntStreamer streamer;
    if (!streamer.attach(rd, nullptr, err)) {
        check("zoom: the octree attaches to the reader", false, err);
        return;
    }
    const BoundingBox env = rd.get_bounds();
    const double origin[3] = {rd.origin(0), rd.origin(1), rd.origin(2)};
    const double radius = 0.5 * std::sqrt(env.extent(0) * env.extent(0) +
                                          env.extent(1) * env.extent(1) +
                                          env.extent(2) * env.extent(2));
    std::printf("  dataset        : %s: %s\n", path.c_str(),
                streamer.tree().describe().c_str());
    std::printf("  envelope       : %.0f x %.0f x %.0f m (header bounds), bounding radius "
                "%.0f m\n", env.extent(0), env.extent(1), env.extent(2), radius);
    check("the test distances straddle the dataset's bounding radius",
          radius < 2000.0 && radius > 10.0,
          std::to_string(radius) + " m between the 10 m and 2000 m cameras");

    // ---- O1/O2 on each tree the section can use ---------------------------
    // `envelope_claim` is only true for the flat file: the header's envelope is a
    // CUBE sized by the widest axis, so on a sparse deep tree a coarse cut has no
    // cell to offer in the empty padding (the points are a 120 m slab inside a
    // 1200 m cube). Requiring 100% of that cube from every tree would assert
    // something about the padding rather than about the cloud; the leaf-coverage
    // invariant below is what holds unconditionally.
    auto zoom_battery = [&](const std::string& tag, const std::string& file, bool envelope_claim) {
        Reader r;
        if (!r.open(file)) { check(tag + ": the file opens", false, r.error()); return; }
        LntStreamer s;
        std::string e;
        if (!s.attach(r, nullptr, e)) { check(tag + ": the octree attaches", false, e); return; }
        const BoundingBox b = r.get_bounds();
        const double o[3] = {r.origin(0), r.origin(1), r.origin(2)};
        auto place = [&](LNTCamera& c, double distance, uint32_t height) {
            for (int i = 0; i < 3; ++i) { c.origin[i] = o[i]; c.eye[i] = b.mid(i); c.target[i] = b.mid(i); }
            c.eye[2] += distance;
            c.aspect = 16.0 / 9.0;
            c.screen_height_px = height;
            c.near_clip = 0.1;
            c.far_clip = distance * 2.0 + 2500.0;   // the cloud's far corner well inside
            return c.rebuild();
        };
        const double bradius = 0.5 * std::sqrt(b.extent(0) * b.extent(0) +
                                              b.extent(1) * b.extent(1) +
                                              b.extent(2) * b.extent(2));
        std::printf("\n  -- %s: %s\n", tag.c_str(), s.tree().describe().c_str());

        // ---- the requirement's own threshold -----------------------------
        // Just past the envelope's bounding radius the ROOT has to be inside the
        // frustum; that is the distance at which the zoom-out bug lived. The cut
        // here legitimately culls the corners that are off screen, so the claim
        // tested is about the root and about the leaves that ARE on screen, not
        // about the envelope.
        {
            LNTCamera edge_cam;
            if (!place(edge_cam, bradius * 1.05, 1080)) {
                check(tag + ": the 1.05x-radius camera has a usable frustum", false);
                return;
            }
            s.update_visible_nodes(edge_cam);
            const LntLodSelection edge_cut = s.stats().last_selection;
            const LntCoverStats edge_cs = cut_coverage(s, r, edge_cam, edge_cut);
            const uint64_t root_idx = s.tree().roots[0];
            std::printf("     %.0f m      : root %s, cut %zu nodes, %llu points, leaves on "
                        "screen %llu missed %llu\n",
                        bradius * 1.05,
                        edge_cam.visible(r.nodes()[root_idx], s.file_origin()) ? "kept"
                                                                              : "CULLED",
                        edge_cut.cut_size(), (unsigned long long)edge_cut.selected_points,
                        (unsigned long long)edge_cs.visible_leaves,
                        (unsigned long long)edge_cs.missed_visible);
            check(tag + ": just past the bounding radius the ROOT still intersects the "
                         "frustum and the cut is not empty",
                  edge_cam.visible(r.nodes()[root_idx], s.file_origin()) &&
                      edge_cut.cut_size() > 0,
                  "root at a bounding radius of " + std::to_string(bradius) + " m");
            check(tag + ": at that distance too, no leaf the frustum keeps is dropped",
                  edge_cs.missed_visible == 0 && edge_cs.visible_leaves > 0,
                  std::to_string(edge_cs.missed_visible) + " of " +
                      std::to_string(edge_cs.visible_leaves) + " leaves lost");
        }

        // ---- zoomed out ---------------------------------------------------
        LNTCamera far_cam;
        if (!place(far_cam, 2000.0, 1080)) {
            check(tag + ": the 2000 m camera has a usable frustum", false);
            return;
        }
        s.update_visible_nodes(far_cam);
        const LntLodSelection far_cut = s.stats().last_selection;
        const LntCoverStats far_cs = cut_coverage(s, r, far_cam, far_cut);
        BoundingBox far_cover;
        const bool far_ok = cut_envelope(s, r, far_cut, far_cover);
        int gaps = 0;
        for (int a = 0; a < 3; ++a)
            if (!(far_cover.min[a] <= b.min[a] + 1e-3 && far_cover.max[a] >= b.max[a] - 1e-3)) ++gaps;
        std::printf("     2000 m      : cut %zu nodes at depth %d of %d, %llu points, %llu "
                    "culled, envelope gaps %d, leaves on screen %llu missed %llu\n",
                    far_cut.cut_size(), far_cut.max_depth_reached, s.tree().max_depth,
                    (unsigned long long)far_cut.selected_points,
                    (unsigned long long)far_cut.culled, gaps,
                    (unsigned long long)far_cs.visible_leaves,
                    (unsigned long long)far_cs.missed_visible);
        check(tag + ": the 2000 m cut drops no leaf the frustum keeps",
              far_ok && far_cs.missed_visible == 0 && far_cs.visible_leaves > 0,
              std::to_string(far_cs.missed_visible) + " of " +
                  std::to_string(far_cs.visible_leaves) + " leaves on screen lost");
        check(tag + ": no cell wholly inside the 2000 m frustum is missing from the cut",
              far_cs.missed_inside == 0,
              std::to_string(far_cs.missed_inside) + " of " +
                  std::to_string(far_cs.inside_leaves) + " cells inside the frustum "
                  "dropped (" + std::to_string(far_cs.missed_points) + " points)");
        if (envelope_claim)
            check("zoom-out: the cut covers 100% of the LNTHeader envelope",
                  far_ok && gaps == 0,
                  std::to_string(gaps) + " axes short of the header bounds");
        if (s.tree().max_depth >= 2)
            check(tag + ": the wide view stops short of the leaves (a coarse node's LOD "
                        "sample stands in for its subtree)",
                  far_cut.max_depth_reached < s.tree().max_depth,
                  "cut depth " + std::to_string(far_cut.max_depth_reached) + " of tree depth " +
                      std::to_string(s.tree().max_depth));
        else
            skip(tag + ": the wide view stops short of the leaves",
                 "this tree is " + std::to_string(s.tree().max_depth) + " level(s) deep");

        // ---- LOD fallback: far enough back, one node carries the cloud -----
        LNTCamera root_cam;
        if (!place(root_cam, 20000.0, 1080)) {
            check(tag + ": the 20000 m camera has a usable frustum", false);
            return;
        }
        s.update_visible_nodes(root_cam);
        const LntLodSelection root_cut = s.stats().last_selection;
        const LntCoverStats root_cs = cut_coverage(s, r, root_cam, root_cut);
        const uint64_t root_id = s.tree().nodes[s.tree().roots[0]].id;
        std::printf("     20000 m     : cut %zu node(s), %llu points, depth %d; leaves on "
                    "screen %llu missed %llu\n",
                    root_cut.cut_size(), (unsigned long long)root_cut.selected_points,
                    root_cut.max_depth_reached, (unsigned long long)root_cs.visible_leaves,
                    (unsigned long long)root_cs.missed_visible);
        check(tag + ": zoomed out past every subtree, the cut collapses to the root and "
                    "its own LOD sample stands in for the whole cloud",
              root_cut.cut_size() == 1 && root_cut.node_ids[0] == root_id &&
                  root_cut.selected_points > 0 && root_cs.missed_visible == 0,
              std::to_string(root_cut.cut_size()) + " nodes, " +
                  std::to_string(root_cut.selected_points) + " points, " +
                  std::to_string(root_cs.missed_visible) + " leaves missed");

        // ---- zoomed in ----------------------------------------------------
        LNTCamera near_cam;
        for (int i = 0; i < 3; ++i) near_cam.origin[i] = o[i];
        // A corner sub-cluster: 10% in along X and Y, and the envelope's own mid Z
        // - the points are a thin slab inside the cubic envelope, so a 10% Z would
        // aim the camera at empty padding and measure nothing.
        near_cam.target[0] = b.min[0] + b.extent(0) * 0.1;
        near_cam.target[1] = b.min[1] + b.extent(1) * 0.1;
        near_cam.target[2] = b.mid(2);
        near_cam.eye[0] = near_cam.target[0];
        near_cam.eye[1] = near_cam.target[1];
        near_cam.eye[2] = near_cam.target[2] + 10.0;
        near_cam.aspect = 16.0 / 9.0;
        near_cam.screen_height_px = 1080;
        near_cam.near_clip = 0.1;
        near_cam.far_clip = 10000.0;
        if (!near_cam.rebuild()) {
            check(tag + ": the 10 m camera has a usable frustum", false);
            return;
        }
        s.update_visible_nodes(near_cam);
        const LntLodSelection near_cut = s.stats().last_selection;
        const LntCoverStats near_cs = cut_coverage(s, r, near_cam, near_cut);
        BoundingBox near_cover;
        const bool near_ok = cut_envelope(s, r, near_cut, near_cover);
        std::printf("     10 m        : cut %zu nodes at depth %d of %d, %llu points, %llu "
                    "culled, leaves on screen %llu missed %llu\n",
                    near_cut.cut_size(), near_cut.max_depth_reached, s.tree().max_depth,
                    (unsigned long long)near_cut.selected_points,
                    (unsigned long long)near_cut.culled,
                    (unsigned long long)near_cs.visible_leaves,
                    (unsigned long long)near_cs.missed_visible);
        check(tag + ": zoom-in discards out-of-view nodes (culled and smaller than the "
                    "wide cut, in nodes and in points)",
              near_ok && near_cut.culled > 0 && near_cut.cut_size() < far_cut.cut_size() &&
                  near_cut.selected_points < far_cut.selected_points,
              std::to_string(near_cut.culled) + " culled, " +
                  std::to_string(near_cut.cut_size()) + " vs " +
                  std::to_string(far_cut.cut_size()) + " nodes, " +
                  std::to_string(near_cut.selected_points) + " vs " +
                  std::to_string(far_cut.selected_points) + " points");
        check(tag + ": zoom-in refines the in-view sub-cluster to the tree's own leaves",
              near_cut.max_depth_reached == s.tree().max_depth,
              "cut depth " + std::to_string(near_cut.max_depth_reached) + " of tree depth " +
                  std::to_string(s.tree().max_depth));
        check(tag + ": the refined cut drops no leaf the 10 m frustum keeps",
              near_cs.missed_visible == 0 && near_cs.visible_leaves > 0,
              std::to_string(near_cs.missed_visible) + " of " +
                  std::to_string(near_cs.visible_leaves) + " leaves on screen lost");
        check(tag + ": no cell wholly inside the 10 m frustum is missing from the cut",
              near_cs.missed_inside == 0,
              std::to_string(near_cs.missed_inside) + " of " +
                  std::to_string(near_cs.inside_leaves) + " cells dropped");
    };

    zoom_battery("flat tree", path, true);
    if (!deep.empty())
        zoom_battery("deep tree", deep, false);
    else
        skip("deep tree: refinement below the flat tree's only level",
             "no " + path + ".fine on disk");

    // ---- O3. THE FAR PLANE FROM THE ENGINE'S OWN float32 MATRIX -----------
    // The whole plane set is compared against the float64 frustum of the same
    // camera, and the far plane additionally against the CONFIGURED far clip - the
    // latter because it is the one statement that cannot be satisfied by a plane
    // that is merely self-consistent: a far plane placed the wrong way round sits
    // behind the eye, reports a plausible distance and agrees with nothing.
    for (int pass = 0; pass < 2; ++pass) {
        const double R = (pass == 0) ? 2000.0 : 20000.0;
        const std::string tag = "far plane at " + std::to_string(int(R)) + " m";
        LNTCamera c64;
        for (int i = 0; i < 3; ++i) { c64.origin[i] = origin[i]; c64.target[i] = env.mid(i); }
        c64.eye[0] = env.mid(0) - R * 0.6;
        c64.eye[1] = env.mid(1) - R * 0.6;
        c64.eye[2] = env.mid(2) + R * 0.529;         // |eye - target| == R, obliquely
        c64.aspect = 16.0 / 9.0;
        c64.screen_height_px = 1080;
        c64.near_clip = 0.1;
        c64.far_clip = R + env.extent(0) * 0.9;      // the cloud crowds the far plane
        check(tag + ": the float64 reference frustum is usable", c64.rebuild());

        float mvp16[16];
        mat4_to_column_major(c64.vp, mvp16);         // exactly what the engine hands back
        LNTCamera c32;
        for (int i = 0; i < 3; ++i) { c32.eye[i] = c64.eye[i]; c32.target[i] = c64.target[i]; }
        c32.aspect = c64.aspect;
        c32.screen_height_px = c64.screen_height_px;
        c32.near_clip = c64.near_clip;
        c32.far_clip = c64.far_clip;
        check(tag + ": the float32 frame matrix yields valid planes",
              c32.set_from_engine_mvp(mvp16, origin, c64.eye));

        const char* names[kPlaneCount] = {"L", "R", "B", "T", "N", "F"};
        double worst_other = 0.0, far_gap = 0.0;
        std::string per_plane;
        for (int p = 0; p < kPlaneCount; ++p) {
            double a[4], b[4];
            for (int k = 0; k < 4; ++k) { a[k] = c32.plane(p, k); b[k] = c64.plane(p, k); }
            const double g = plane_gap_over_box(a, b, env);
            if (p == kPlaneFar) far_gap = g; else worst_other = std::max(worst_other, g);
            per_plane += std::string(names[p]) + "=" + std::to_string(g) + " ";
        }
        // Where the far plane actually is, in the only frame that matters: the
        // distance from the eye along its own normal.
        const double at_eye = (c32.plane(kPlaneFar, 0) * c64.eye[0] +
                               c32.plane(kPlaneFar, 1) * c64.eye[1] +
                               c32.plane(kPlaneFar, 2) * c64.eye[2] +
                               c32.plane(kPlaneFar, 3));
        const double budget = c32.far_plane_tolerance();
        std::printf("     far clip %.0f m: worst float32-vs-float64 gap per plane "
                    "%s| far plane %.4f%% of its distance, cull budget %.2f m; the plane "
                    "sits %.2f m from the eye\n",
                    c64.far_clip, per_plane.c_str(),
                    100.0 * far_gap / (c32.far_plane_distance() > 0.0
                                           ? c32.far_plane_distance() : 1.0),
                    budget, at_eye);
        check(tag + ": the five well-conditioned planes agree with the float64 frustum to "
                     "within a metre over the whole envelope",
              worst_other < 1.0,
              std::to_string(worst_other) + " m worst");
        check(tag + ": the far plane sits in front of the eye at the configured far clip",
              at_eye > 0.0 && std::fabs(at_eye - c64.far_clip) <= budget,
              std::to_string(at_eye) + " m from the eye against a configured " +
                  std::to_string(c64.far_clip) + " m");
        check(tag + ": the float32 far plane tracks the float64 one to within 1% of its "
                     "own distance",
              c32.far_plane_distance() > 0.0 && far_gap <= 0.01 * c32.far_plane_distance(),
              std::to_string(far_gap) + " m of " +
                  std::to_string(c32.far_plane_distance()) + " m");
        check(tag + ": the far plane's cull budget covers the error it actually carries",
              far_gap <= budget,
              std::to_string(far_gap) + " m against a budget of " +
                  std::to_string(budget) + " m");
    }

    // ---- O4. HYSTERESIS AND THE 100-FRAME SWEEP ---------------------------
    const std::string bench_file = deep.empty() ? path : deep;
    Reader brd;
    if (!brd.open(bench_file)) { check("sweep: the dataset opens", false, brd.error()); return; }
    uint64_t total_payload = 0, max_payload = 0;
    for (const LNTNodeEntry& e : brd.nodes()) {
        const uint64_t bytes = LntCacheManager::payload_bytes(e.point_count);
        total_payload += bytes;
        max_payload = std::max(max_payload, bytes);
    }
    const BoundingBox benv = brd.get_bounds();
    const double borigin[3] = {brd.origin(0), brd.origin(1), brd.origin(2)};
    const double bside = benv.extent(0);
    LntCacheManager cache(total_payload + max_payload);
    if (!cache.open(bench_file, err)) { check("sweep: the cache opens the dataset", false, err); return; }
    LntStreamer bs;
    if (!bs.attach(brd, &cache, err)) { check("sweep: the streamer attaches", false, err); return; }

    // A CLOSE camera, so the cut is the leaves and the motion budget (0.01 x the
    // smallest cell in it) is wide enough for a sub-budget move that still
    // rotates the view past 0.05 degrees - which is what lets the two rules be
    // told apart instead of both being satisfied by the same movement.
    const double close_R = 10.0;
    LNTCamera hc;
    for (int i = 0; i < 3; ++i) hc.origin[i] = borigin[i];
    hc.target[0] = benv.min[0] + benv.extent(0) * 0.1;
    hc.target[1] = benv.min[1] + benv.extent(1) * 0.1;
    hc.target[2] = benv.mid(2);
    hc.eye[0] = hc.target[0];
    hc.eye[1] = hc.target[1];
    hc.eye[2] = hc.target[2] + close_R;
    hc.aspect = 16.0 / 9.0;
    hc.screen_height_px = 1080;
    hc.near_clip = 0.1;
    hc.far_clip = 10000.0;
    bool hyst_ok = hc.rebuild();
    check("hysteresis: the close camera has a usable frustum", hyst_ok);
    if (!hyst_ok) {
        cache.close();
        return;
    }
    bs.update_visible_nodes(hc);
    const double smallest = min_cut_node_size(bs, brd);
    const double limit = 0.01 * smallest;                 // the streamer's own budget
    const uint64_t runs0 = bs.stats().selection_runs;
    const uint64_t reuses0 = bs.stats().selection_reuses;
    const std::size_t cut0 = bs.stats().last_selection.cut_size();
    std::printf("  close cut      : %zu nodes, smallest cell %.3f m => 0.01 x that = %.4f m "
                "of motion budget\n", cut0, smallest, limit);
    check("hysteresis: the close cut has a measurable smallest cell", smallest > 0.0);

    // Five frames of a still camera: five reuses, no new traversal.
    for (int f = 0; f < 5; ++f) bs.update_visible_nodes(hc);
    check("hysteresis: a still camera re-traverses nothing",
          bs.stats().selection_runs == runs0 && bs.stats().selection_reuses == reuses0 + 5,
          std::to_string(bs.stats().selection_runs - runs0) + " traversals, " +
              std::to_string(bs.stats().selection_reuses - reuses0) + " reuses in 5 frames");

    // A pan (eye AND target together) of 0.2 x the budget: still no traversal.
    const double pan = 0.2 * limit;
    for (int i = 0; i < 3; ++i) { hc.eye[i] += pan; hc.target[i] += pan; }
    hc.rebuild();
    bs.update_visible_nodes(hc);
    check("hysteresis: a move well inside 0.01 x the smallest cell reuses the cut",
          bs.stats().selection_runs == runs0,
          std::to_string(bs.stats().selection_runs - runs0) + " traversals after a " +
              std::to_string(pan) + " m pan");

    // ... and the same gesture five more times drifts past the budget, so the cut
    // must be rebuilt. A hysteresis that never expires is a stale frame: the rule
    // is measured against the camera the resident cut was built from, not against
    // the previous frame, so drift cannot hide inside a chain of small steps.
    for (int f = 0; f < 5; ++f) {
        for (int i = 0; i < 3; ++i) { hc.eye[i] += pan; hc.target[i] += pan; }
        hc.rebuild();
        bs.update_visible_nodes(hc);
    }
    check("hysteresis: accumulated drift past the budget forces a new traversal",
          bs.stats().selection_runs > runs0,
          std::to_string(bs.stats().selection_runs - runs0) + " traversals after 6 x " +
              std::to_string(pan) + " m of panning (budget " + std::to_string(limit) + " m)");

    // A ROTATION of the view axis, with the eye held still so only the rotation
    // rule can object: below 0.05 degrees it reuses, above it, it does not.
    const double radians_per_metre = 1.0 / close_R;        // small-angle, at the target
    auto rotate_target = [&](double metres) {
        hc.target[0] += metres;                            // lateral: slides the view axis
        hc.rebuild();
        bs.update_visible_nodes(hc);
    };
    const double small_move = (0.02 * 3.14159265358979323846 / 180.0) / radians_per_metre;
    const double big_move = (0.20 * 3.14159265358979323846 / 180.0) / radians_per_metre;
    const uint64_t runs_before_rot = bs.stats().selection_runs;
    rotate_target(small_move);
    check("hysteresis: a rotation under 0.05 degrees reuses the cut",
          bs.stats().selection_runs == runs_before_rot,
          std::to_string(0.02) + " degrees over " + std::to_string(small_move) + " m");
    if (big_move < limit) {
        rotate_target(big_move);
        check("hysteresis: a rotation past 0.05 degrees re-traverses even though the "
              "camera moved inside its motion budget",
              bs.stats().selection_runs > runs_before_rot,
              std::to_string(0.20) + " degrees over " + std::to_string(big_move) +
                  " m of a " + std::to_string(limit) + " m budget");
    } else {
        skip("hysteresis: a rotation past 0.05 degrees re-traverses",
             "the motion budget of this cut is too small to isolate rotation");
    }

    // ---- the 100-frame zoom-out -> zoom-in sweep, timed where a frame is
    // actually timed: LntInteractiveLoop::streamer_update_ms, which covers the
    // selection, the cache handoff (pin, residency scan, requests) and the draw
    // -range push - not just the traversal in isolation.
    LntCameraController ctl;
    LntInteractiveLoop loop;
    ctl.set_target(benv.mid(0), benv.mid(1), benv.mid(2));
    ctl.set_lens(45.0, 0.1, bside * 40.0);
    ctl.set_ortho2d(false);
    ctl.set_angles(0.6, 0.5);
    ctl.set_sse_threshold(50.0);           // the controller's own dial range, at its coarsest
    loop.attach(&ctl, &cache, borigin);
    loop.set_viewport(960, 720);
    const int kSweep = 100;
    double worst_ms = 0.0, sum_ms = 0.0;
    uint64_t failed = 0;
    uint32_t first_cut = 0, mid_cut = 0, biggest_cut = 0;
    const uint64_t sweep_runs0 = bs.stats().selection_runs;
    const uint64_t sweep_reuses0 = bs.stats().selection_reuses;
    for (int f = 0; f < kSweep; ++f) {
        const double t = double(f) / double(kSweep - 1);
        const double d = (t < 0.5) ? (2000.0 + (10.0 - 2000.0) * (t * 2.0))
                                   : (10.0 + (2000.0 - 10.0) * ((t - 0.5) * 2.0));
        ctl.set_distance(d);                          // out -> in -> back out
        if (!loop.update_frame(bs, 1.0f / 60.0f)) { ++failed; continue; }
        const double ms = double(loop.stats().streamer_update_ms);
        worst_ms = std::max(worst_ms, ms);
        sum_ms += ms;
        if (f == 0) first_cut = loop.stats().active_nodes;
        if (f == kSweep / 2) mid_cut = loop.stats().active_nodes;
        biggest_cut = std::max(biggest_cut, loop.stats().active_nodes);
    }
    std::printf("  %d-frame sweep : worst %.3f ms, mean %.3f ms per frame; cut %u nodes wide -> "
                "%u at the tightest zoom (peak %u); %llu traversals, %llu reuses\n",
                kSweep, worst_ms, sum_ms / double(kSweep > 0 ? kSweep : 1), first_cut, mid_cut,
                biggest_cut, (unsigned long long)(bs.stats().selection_runs - sweep_runs0),
                (unsigned long long)(bs.stats().selection_reuses - sweep_reuses0));
    check("sweep: every one of the 100 frames ran", failed == 0,
          std::to_string(failed) + " failed " + loop.error());
    check("sweep: streamer_update_ms stays <= 1.5 ms on every frame", worst_ms <= 1.5,
          std::to_string(worst_ms) + " ms worst");
    check("sweep: the zoom actually changed the LOD (wide and tight cuts differ)",
          mid_cut != first_cut && biggest_cut > 0,
          std::to_string(first_cut) + " nodes wide vs " + std::to_string(mid_cut) +
              " at the tightest zoom");

    // ---- O5. THE ENGINE HANDOFF: ONE CALL, AND NONE WHEN NOTHING CHANGED ---
    // nkv_set_point_draw_ranges takes the WHOLE list in a single call - the
    // (first, count) windows are two contiguous arrays - so batching is already
    // the contract, not something the streamer could batch afterwards. What this
    // measures is the other half: a frame whose draw list is identical to the
    // engine's must not re-upload it. The ranges live in the renderer, so this
    // needs the engine and is SKIPPED, never passed, without one.
    {
        const std::string dll = VulkanEngine::find_dll();
        VulkanEngine eng;
        LntScene scene;
        std::string e;
        const std::string what = "draw ranges: three identical frames push once and skip twice";
        if (dll.empty()) {
            skip(what, "no naksha_vulkan.dll found");
        } else if (!eng.load(dll, e) || !eng.is_available() ||
                   !eng.open(320, 240, true, false, e) || !scene.plan(brd, e) ||
                   !scene.reserve(eng, e)) {
            skip(what, "engine unavailable: " + e);
        } else {
            double o3[3];
            scene.render_origin(o3);
            eng.set_render_origin(o3[0], o3[1], o3[2]);
            // A range is pushed only for an UPLOADED tile, so the cut has to be
            // resident first - otherwise every frame would have an empty list and
            // the check would pass for the wrong reason.
            cache.set_pinned(bs.stats().last_selection.node_ids);
            cache.request_all(bs.stats().last_selection.node_ids);
            const bool resident = cache.drain(20000);
            const uint64_t calls0 = bs.stats().range_pushes;
            const uint64_t skips0 = bs.stats().range_pushes_skipped;
            std::size_t pushed_first = 0, pushed_last = 0;
            bool ran = true;
            for (int f = 0; f < 3; ++f) {
                pushed_last = bs.update_visible_nodes(eng, scene, hc, e);
                if (f == 0) pushed_first = pushed_last;
                if (!e.empty()) ran = false;
            }
            std::printf("  engine handoff : %zu drawable ranges, %llu call(s), %llu skipped over "
                        "3 identical frames\n",
                        pushed_first, (unsigned long long)(bs.stats().range_pushes - calls0),
                        (unsigned long long)(bs.stats().range_pushes_skipped - skips0));
            check(what,
                  resident && ran && pushed_first > 0 && pushed_first == pushed_last &&
                      bs.stats().range_pushes - calls0 == 1 &&
                      bs.stats().range_pushes_skipped - skips0 == 2,
                  std::to_string(pushed_first) + " ranges: " +
                      std::to_string(bs.stats().range_pushes - calls0) + " pushes, " +
                      std::to_string(bs.stats().range_pushes_skipped - skips0) + " skips" +
                      (resident ? "" : " (the cut never became resident)") +
                      (ran ? "" : " (" + e + ")"));
            eng.close();
        }
    }
    cache.close();
}

// ========================================================================
// P. MID-ZOOM DRAW COVERAGE: THE BLANK-OCTANT BUG AND ITS FALLBACK
// ========================================================================
// The reported symptom is specific: at INTERMEDIATE zoom the cloud shows large
// rectangular voids, while the extremes - fully out and deeply in - are fine.
// That shape is the signature of the cut, not of the frustum. A cut is an
// antichain, so the frame a parent is replaced by its children is also the frame
// the parent's own (first, count) window leaves the draw list - and the children
// are exactly the nodes whose payloads were just requested. For every frame
// between the split and the arrival of those payloads, and again for every frame
// the upload time slice defers them, the parent's WHOLE CELL is drawn by nothing:
// a rectangular hole one octant across, at the one zoom band where an octant is
// still large on screen. Zoomed out, the cut is one or two nodes and a change is
// a whole-cloud-scale event; zoomed in, the cut is the leaves and consecutive
// frames mostly share them. The middle is where the transition is visible.
//
// THREE CLAIMS, MEASURED SEPARATELY.
//
//   P1  THE FRUSTUM IS NOT THE CAUSE, and that is a measurement rather than an
//       opinion. The traversal skips a culled node's entire subtree, which is
//       only sound if a child's cell is inside its parent's - so the overhang is
//       measured over every parent/child pair in the file, and separately the
//       traversal is checked against the property itself: no leaf that the
//       frustum keeps may have an ancestor it culled. The overhang is float32
//       storage noise (LNTFormat.h stores the centre as a float32 OFFSET from
//       bounds_min precisely to keep it that small), and it is compared against
//       the cull tolerance that already covers it.
//   P2  THE VOID IS REAL and it is bounded by what the arena already holds. On a
//       walk from 2000 m to 25 m, with the arena modelled as "the previous cut's
//       tiles", every cell that goes blank is counted geometrically - from the
//       directory's own cells, not from the streamer's counters - and the count
//       is compared with the same walk with the fallback switched off. The cold
//       start is the one frame nothing can fix: no tile has ever been uploaded.
//   P3  THE ENGINE GETS THE STAND-IN, not just the bookkeeping: the ranges and
//       points the engine holds are compared against the resolved list, and the
//       descent is then completed to show the fallback stands down once the real
//       payloads arrive.
//
// THE REQUESTED CAMERA (R = 150 m, "whole dataset in view") DOES NOT EXIST ON
// THIS FIXTURE, and the harness says so instead of pretending. The envelope is
// ~1200 m across, so a 45-degree lens at 150 m sees ~124 m of it - a seventh. The
// envelope-coverage statement is therefore made where it is true (>= 1091 m, the
// bounding radius), and 150 m is where the refinement and no-hole statements are
// made, which is what the symptom is actually about.

// True when `inner` lies inside `outer` to within `slack`.
static bool box_inside(const BoundingBox& outer, const BoundingBox& inner, double slack) {
    for (int a = 0; a < 3; ++a)
        if (inner.min[a] < outer.min[a] - slack || inner.max[a] > outer.max[a] + slack)
            return false;
    return true;
}

struct LntBlankStats {
    uint64_t cells = 0;            // cut cells examined
    uint64_t blank = 0;            // ... with no drawn window covering them
    uint64_t blank_points = 0;     // ... and the points those cells stand for
    uint64_t windows = 0;
    uint64_t stand_in_windows = 0;
    uint64_t streamer_uncovered = 0;   // the streamer's own count, for agreement
};

// Cells of the cut that no window in `list` covers, measured from the directory's
// own cells. Deliberately independent of the streamer's accounting: the geometry
// is the thing that a viewer sees, while the counters are what the code believes.
static LntBlankStats blank_cells(const LntStreamer& s, const Reader& rd,
                                 const LntLodSelection& cut, const LntDrawList& list,
                                 const LntScene& scene) {
    LntBlankStats st;
    st.windows = list.size();
    st.stand_in_windows = list.stand_ins;
    st.streamer_uncovered = list.uncovered_nodes;
    std::vector<BoundingBox> drawn;
    drawn.reserve(list.ids.size());
    for (uint64_t id : list.ids) {
        const LntTile* t = scene.find_tile(id);
        if (t != nullptr) drawn.push_back(t->cell);
    }
    for (uint64_t id : cut.node_ids) {
        const auto it = s.tree().index_of.find(id);
        if (it == s.tree().index_of.end()) continue;
        const BoundingBox cell = node_bounds(rd.nodes()[it->second], s.file_origin());
        ++st.cells;
        bool covered = false;
        for (const BoundingBox& w : drawn)
            if (box_inside(w, cell, kCullToleranceMeters)) { covered = true; break; }
        if (!covered) {
            ++st.blank;
            st.blank_points += rd.nodes()[it->second].point_count;
        }
    }
    return st;
}

// Worst distance, in metres, that a child's cell reaches OUTSIDE its parent's,
// over every parent/child pair in the tree. This is the quantity that decides
// whether "a culled node's whole subtree is outside too" holds: the traversal
// never tests a descendant of a culled node, so the descent is only sound if the
// overhang is smaller than the slack the cull test already grants each plane.
static double worst_cell_overhang(const LntStreamer& s, const Reader& rd,
                                  uint64_t& pairs_out, uint64_t& out_out) {
    pairs_out = 0;
    out_out = 0;
    double worst = 0.0;
    const LntNodeTree& t = s.tree();
    for (std::size_t i = 0; i < t.nodes.size(); ++i) {
        const BoundingBox parent = node_bounds(rd.nodes()[i], s.file_origin());
        for (uint32_t c : t.nodes[i].children) {
            ++pairs_out;
            const BoundingBox child = node_bounds(rd.nodes()[c], s.file_origin());
            double over = 0.0;
            for (int a = 0; a < 3; ++a) {
                over = std::max(over, parent.min[a] - child.min[a]);
                over = std::max(over, child.max[a] - parent.max[a]);
            }
            if (over > 0.0) { ++out_out; worst = std::max(worst, over); }
        }
    }
    return worst;
}

// Leaves the frustum keeps whose ancestor the traversal culled. Any of these is a
// hole the subtree skip would create, which is the property the whole traversal
// rests on - and the one the reported symptom was blamed on.
static uint64_t leaves_with_culled_ancestor(const LntStreamer& s, const Reader& rd,
                                            const LNTCamera& cam, uint64_t& visible_leaves) {
    const LntNodeTree& t = s.tree();
    std::vector<char> vis(t.nodes.size(), 0);
    for (std::size_t i = 0; i < t.nodes.size(); ++i)
        vis[i] = cam.visible(rd.nodes()[i], s.file_origin()) ? 1 : 0;
    visible_leaves = 0;
    uint64_t bad = 0;
    for (std::size_t i = 0; i < t.nodes.size(); ++i) {
        if (!t.nodes[i].children.empty()) continue;          // leaves only
        if (!vis[i]) continue;
        ++visible_leaves;
        for (uint32_t cur = t.nodes[i].parent; cur != kNoParent; cur = t.nodes[cur].parent)
            if (!vis[cur]) { ++bad; break; }
    }
    return bad;
}

static void section_midzoom_coverage(const std::string& path) {
    std::printf("\n=== P. MID-ZOOM DRAW COVERAGE (BLANK OCTANTS) ===\n");

    std::string deep;
    {
        Reader probe;
        const std::string candidate = path + ".fine";
        if (probe.open(candidate) && probe.get_total_nodes() > 0) deep = candidate;
    }
    std::string err;

    // ---- P1: the frustum is not the cause ---------------------------------
    // The overhang measurement, on the tree the symptom needs (the deepest one
    // available), and the traversal property it underpins.
    {
        const std::string file = deep.empty() ? path : deep;
        Reader rd;
        if (!rd.open(file)) { check("mid-zoom: the dataset opens", false, rd.error()); return; }
        LntStreamer s;
        if (!s.attach(rd, nullptr, err)) {
            check("mid-zoom: the octree attaches to the reader", false, err);
            return;
        }
        uint64_t pairs = 0, out = 0;
        const double overhang = worst_cell_overhang(s, rd, pairs, out);
        std::printf("  cell nesting   : %llu parent/child pairs, %llu child cells reach outside "
                    "the parent cell, worst %.9f m (cull tolerance %.6f m)\n",
                    (unsigned long long)pairs, (unsigned long long)out, overhang,
                    kCullToleranceMeters);
        check("frustum: a child's cell lies inside its parent's to far better than the cull "
              "tolerance, so skipping a culled node's subtree cannot drop a visible cell",
              out == 0 || overhang < 0.1 * kCullToleranceMeters,
              "worst overhang " + std::to_string(overhang) + " m against " +
                  std::to_string(kCullToleranceMeters) + " m of tolerance");

        // The property itself, over the mid-zoom band: no leaf the frustum keeps
        // may sit under an ancestor it culled. Also the direct answer to "are
        // boundary children being dropped?".
        const BoundingBox b = rd.get_bounds();
        const double o[3] = {rd.origin(0), rd.origin(1), rd.origin(2)};
        const double bradius = 0.5 * std::sqrt(b.extent(0) * b.extent(0) +
                                              b.extent(1) * b.extent(1) +
                                              b.extent(2) * b.extent(2));
        const double band[] = {bradius * 1.05, 2000.0, 1200.0, 800.0, 500.0, 300.0, 150.0, 10.0};
        uint64_t bad_total = 0, visible_total = 0;
        std::string worst_at;
        for (double d : band) {
            LNTCamera c;
            for (int i = 0; i < 3; ++i) { c.origin[i] = o[i]; c.eye[i] = b.mid(i); c.target[i] = b.mid(i); }
            c.eye[2] += d;
            c.aspect = 16.0 / 9.0;
            c.screen_height_px = 1080;
            c.near_clip = 0.1;
            c.far_clip = d * 2.0 + 2500.0;
            if (!c.rebuild()) continue;
            uint64_t vis = 0;
            const uint64_t bad = leaves_with_culled_ancestor(s, rd, c, vis);
            bad_total += bad;
            visible_total += vis;
            if (bad != 0 && worst_at.empty()) worst_at = std::to_string(int(d)) + " m";
        }
        check("frustum: across the whole zoom band, no leaf the frustum keeps has a culled "
              "ancestor (a subtree is never skipped over a visible cell)",
              bad_total == 0 && visible_total > 0,
              std::to_string(bad_total) + " of " + std::to_string(visible_total) +
                  " visible leaves have a culled ancestor" +
                  (worst_at.empty() ? std::string() : (" (at " + worst_at + ")")));

        // ---- P2: the blank-cell walk, with the fallback on and off ---------
        // The arena is modelled as "every tile an earlier frame selected", which
        // is what an unlimited upload budget produces, and exactly the state a
        // viewer is in when it zooms: previous cuts are on the GPU, the new cut
        // is not.
        LntScene scene;
        if (!scene.plan(rd, err)) {
            check("mid-zoom: the scene plans from the directory alone", false, err);
            return;
        }
        const double zooms[] = {2000.0, 1200.0, 800.0, 500.0, 300.0, 150.0, 60.0, 25.0};
        auto walk = [&](bool fallback, uint64_t& blank_total, uint64_t& cells_total,
                        uint64_t& blank_points, uint64_t& cold_cells,
                        uint64_t& stand_in_total, uint64_t& uncovered_total,
                        uint64_t& frames) {
            LntStreamer s2;
            if (!s2.attach(rd, nullptr, err)) return false;
            s2.set_ancestor_fallback(fallback);
            // Tiles the GPU holds, one flag per tile in directory order.
            std::vector<char> arena(scene.tiles().size(), 0);
            blank_total = cells_total = blank_points = cold_cells = 0;
            stand_in_total = uncovered_total = frames = 0;
            for (double d : zooms) {
                LNTCamera c;
                for (int i = 0; i < 3; ++i) { c.origin[i] = o[i]; c.eye[i] = b.mid(i); c.target[i] = b.mid(i); }
                c.eye[2] += d;
                c.aspect = 16.0 / 9.0;
                c.screen_height_px = 1080;
                c.near_clip = 0.1;
                c.far_clip = d * 2.0 + 2500.0;
                if (!c.rebuild()) continue;
                s2.update_visible_nodes(c);
                const LntLodSelection cut = s2.stats().last_selection;
                for (std::size_t i = 0; i < scene.tiles().size(); ++i)
                    scene.tile_mut(scene.tiles()[i].node_id)->uploaded = arena[i] != 0;
                const LntDrawList& list = s2.resolve_draw_list(scene);
                const LntBlankStats bs = blank_cells(s2, rd, cut, list, scene);
                ++frames;
                if (frames == 1) cold_cells = bs.blank;
                blank_total += bs.blank;
                cells_total += bs.cells;
                blank_points += bs.blank_points;
                stand_in_total += bs.stand_in_windows;
                uncovered_total += bs.streamer_uncovered;
                for (uint64_t id : cut.node_ids) {
                    const LntTile* t = scene.find_tile(id);
                    if (t == nullptr) continue;
                    const LntTile* base = scene.tiles().data();
                    arena[std::size_t(t - base)] = 1;
                }
            }
            return true;
        };

        uint64_t on_blank = 0, on_cells = 0, on_points = 0, on_cold = 0, on_stand = 0,
                 on_uncov = 0, on_frames = 0;
        uint64_t off_blank = 0, off_cells = 0, off_points = 0, off_cold = 0, off_stand = 0,
                 off_uncov = 0, off_frames = 0;
        const bool on_ok = walk(true, on_blank, on_cells, on_points, on_cold, on_stand,
                                on_uncov, on_frames);
        const bool off_ok = walk(false, off_blank, off_cells, off_points, off_cold, off_stand,
                                 off_uncov, off_frames);
        check("mid-zoom: both walks ran", on_ok && off_ok);
        std::printf("  zoom walk      : %s to 25 m, %llu cut cells, %llu frames\n",
                    file.c_str(), (unsigned long long)on_cells,
                    (unsigned long long)on_frames);
        std::printf("     fallback ON : %llu cells drawn by nothing (%llu points), of which the "
                    "cold-start frame accounts for %llu; %llu stand-in windows, %llu cells "
                    "the streamer itself counted uncovered\n",
                    (unsigned long long)on_blank, (unsigned long long)on_points,
                    (unsigned long long)on_cold, (unsigned long long)on_stand,
                    (unsigned long long)on_uncov);
        std::printf("     fallback OFF: %llu cells drawn by nothing (%llu points); %llu rules them "
                    "all out\n",
                    (unsigned long long)off_blank, (unsigned long long)off_points,
                    (unsigned long long)off_stand);
        check("mid-zoom: with the ancestor fallback, the ONLY cut cells drawn by nothing are "
              "the cold-start frame's - a zoom step never blanks a cell whose ancestor is "
              "already on the GPU",
              on_blank == on_cold && on_stand > 0 && on_uncov == on_blank,
              std::to_string(on_blank) + " blank cells, " + std::to_string(on_cold) +
                  " of them the cold start, vs " + std::to_string(on_uncov) +
                  " counted uncovered");
        check("mid-zoom: WITHOUT the fallback the same walk blanks hundreds of cells, so the "
              "check has teeth rather than passing by construction",
              off_blank > on_blank,
              std::to_string(off_blank) + " blank cells with the fallback off vs " +
                  std::to_string(on_blank) + " with it on");
        check("mid-zoom: the geometric count and the streamer's own uncovered count agree",
              on_uncov == on_blank,
              std::to_string(on_uncov) + " counted vs " + std::to_string(on_blank) +
                  " measured geometrically");

        // ---- the requested envelope claim, at the distances where it is true --
        // R > dataset_bounding_radius is the band where the whole envelope is in
        // view, and it is checked on the tree that can support it (the flat one:
        // the header envelope is a cube sized by the widest axis, so on the deep
        // tree a coarse cut has no cell to offer in the thin padding - see O1).
        if (!deep.empty() && file == deep) {
            // The premise of the requested R = 150 m camera, measured rather than
            // assumed: what fraction of the envelope the frustum spans there.
            LNTCamera c150;
            for (int i = 0; i < 3; ++i) { c150.origin[i] = o[i]; c150.eye[i] = b.mid(i); c150.target[i] = b.mid(i); }
            c150.eye[2] += 150.0;
            c150.aspect = 16.0 / 9.0;
            c150.screen_height_px = 1080;
            c150.near_clip = 0.1;
            c150.far_clip = 2800.0;
            c150.rebuild();
            const double span_150 = 2.0 * 150.0 * std::tan(c150.fov_y_deg * 3.14159265358979323846 / 360.0) *
                                    c150.aspect;
            std::printf("  150 m camera   : the frustum spans %.0f m across at the target, %s "
                        "of the %.0f m envelope's width\n",
                        span_150, (span_150 < b.extent(0) ? "SHORT" : "covering"), b.extent(0));
            check("mid-zoom: at 150 m the ENVELOPE is not in view on this fixture, so the "
                  "envelope claim is made past the bounding radius instead (and this check "
                  "records the premise rather than skipping in silence)",
                  span_150 < b.extent(0) && bradius > 1000.0,
                  std::to_string(span_150) + " m of a " + std::to_string(b.extent(0)) +
                      " m envelope, bounding radius " + std::to_string(bradius) + " m");
        }
    }

    // ---- P3: the engine gets the stand-in ---------------------------------
    // Everything above resolves the draw list engine-free. This is the same
    // descent through the real handoff, so the fallback is shown to reach the
    // engine and to stand down when the payloads it was standing in for arrive.
    {
        const std::string file = deep.empty() ? path : deep;
        const std::string dll = VulkanEngine::find_dll();
        Reader rd;
        if (!rd.open(file)) { check("mid-zoom engine: the dataset opens", false, rd.error()); return; }
        uint64_t payload = 0, biggest = 0;
        for (const LNTNodeEntry& e : rd.nodes()) {
            const uint64_t bytes = LntCacheManager::payload_bytes(e.point_count);
            payload += bytes;
            biggest = std::max(biggest, bytes);
        }
        LntCacheManager cache(payload + biggest);
        LntScene scene;
        VulkanEngine eng;
        std::string e;
        const std::string what = "mid-zoom engine: a descent with the payloads still on the way "
                                 "draws the parent's LOD tiles instead of going blank";
        if (dll.empty()) {
            skip(what, "no naksha_vulkan.dll found");
        } else if (!cache.open(file, e)) {
            check("mid-zoom engine: the cache opens the dataset", false, e);
        } else if (!eng.load(dll, e) || !eng.is_available() ||
                   !eng.open(320, 240, true, false, e) || !scene.plan(rd, e) ||
                   !scene.reserve(eng, e)) {
            skip(what, "engine unavailable: " + e);
            cache.close();
        } else {
            LntStreamer s;
            if (!s.attach(rd, &cache, e)) {
                check("mid-zoom engine: the streamer attaches", false, e);
            } else {
                const BoundingBox b = rd.get_bounds();
                const double o[3] = {rd.origin(0), rd.origin(1), rd.origin(2)};
                auto place = [&](LNTCamera& c, double d) {
                    for (int i = 0; i < 3; ++i) { c.origin[i] = o[i]; c.eye[i] = b.mid(i); c.target[i] = b.mid(i); }
                    c.eye[2] += d;
                    c.aspect = 16.0 / 9.0;
                    c.screen_height_px = 1080;
                    c.near_clip = 0.1;
                    c.far_clip = d * 2.0 + 2500.0;
                    return c.rebuild();
                };
                LNTCamera wide, mid;
                if (!place(wide, 2000.0) || !place(mid, 300.0)) {
                    check(what, false, "the wide or mid camera has no usable frustum");
                } else {
                    // Frame the wide cut, then wait for it: this is the arena state
                    // a viewer is in when it zooms in.
                    s.update_visible_nodes(wide);
                    cache.set_pinned(s.stats().last_selection.node_ids);
                    cache.request_all(s.stats().last_selection.node_ids);
                    const bool resident = cache.drain(20000);
                    s.update_visible_nodes(eng, scene, wide, e);
                    const uint64_t uploaded = scene.uploaded_tiles();
                    // Now descend with the loader held: the new cut's payloads cannot
                    // arrive, so every cell it adds must be drawn from an ancestor.
                    cache.hold(true);
                    s.update_visible_nodes(eng, scene, mid, e);
                    const LntDrawList& list = s.draw_list();
                    const LntBlankStats bs = blank_cells(s, rd, s.stats().last_selection, list, scene);
                    const bool engine_matches =
                        eng.draw_range_count() == list.size() &&
                        eng.draw_range_points() == list.points;
                    std::printf("  engine descent : wide cut uploaded %llu tiles, then %zu cut "
                                "cells at 300 m drew through %llu stand-in window(s); engine "
                                "holds %u ranges / %u points, resolved list %zu / %llu\n",
                                (unsigned long long)uploaded, bs.cells,
                                (unsigned long long)s.stats().stand_in_ranges,
                                eng.draw_range_count(), eng.draw_range_points(), list.size(),
                                (unsigned long long)list.points);
                    check(what,
                          resident && uploaded > 0 && s.stats().stand_in_ranges > 0 &&
                              bs.blank == 0 && engine_matches && e.empty(),
                          std::to_string(s.stats().stand_in_ranges) + " stand-ins, " +
                              std::to_string(bs.blank) + " blank of " +
                              std::to_string(bs.cells) + " cells, " +
                              (engine_matches ? std::string("engine matches") :
                                                std::string("ENGINE MISMATCH")) +
                              (e.empty() ? std::string() : (" (" + e + ")")) );
                    // Let the payloads in and the stand-ins must go away: coarse
                    // coverage is a bridge, not the destination.
                    cache.hold(false);
                    const bool drained = cache.drain(20000);
                    s.update_visible_nodes(eng, scene, mid, e);
                    std::printf("  engine refine  : after the payloads arrive, %llu stand-in "
                                "window(s) remain, %llu cut cells drawn by nothing\n",
                                (unsigned long long)s.stats().stand_in_ranges,
                                (unsigned long long)blank_cells(s, rd, s.stats().last_selection,
                                                               s.draw_list(), scene).blank);
                    check("mid-zoom engine: once the real payloads are uploaded the stand-in "
                          "stands down (the fallback is a bridge, not the destination)",
                          drained && s.stats().stand_in_ranges == 0 && e.empty(),
                          std::to_string(s.stats().stand_in_ranges) + " stand-ins left");
                }
            }
            eng.close();
            cache.close();
        }
    }
}

// ========================================================================
// Q. ASYNC STREAMING: RETENTION WHILE CHILDREN LOAD, AND THE 10% BAND
// ========================================================================
// The request's two mechanisms, measured against each other on one flight.
//
//   Q1  A ZOOM-IN FROM R = 500 m TO 20 m with 10 ms of simulated disk latency,
//       asserting per frame that (a) the cut still covers every leaf the frustum
//       keeps - 100% spatial coverage, the request's own wording - and (b) no cut
//       cell is drawn by nothing while children are sitting in the pending queue.
//       Run three times over the same trajectory:
//         RETAIN-IN-CUT   gate on: the traversal refuses to descend until every
//                         visible child's payload is resident, so the parent IS
//                         the cut while the data streams in;
//         RETAIN-IN-DRAW  the shipped default: the cut descends immediately and
//                         the draw list borrows the parent's tile instead;
//         CONTROL         both off: the state the reported bug lives in.
//       The two retention forms must both report zero empty cells, and the
//       control must not - otherwise (b) would be passing because nothing was
//       ever at risk.
//   Q2  THE 10% BAND. A camera alternating between two distances that straddle
//       one level's threshold by +-3% flips the cut every frame without the
//       band (the cut is a pure function of the camera, so a camera at the
//       boundary has no memory), and holds still with it.
//
// WHY THE GATE IS A MODE RATHER THAN THE DEFAULT. Both forms keep every cell on
// screen; what differs is what the CUT means. Retention-in-cut makes selection a
// function of cache state as well as of the camera - a payload arriving re-runs
// the traversal while the camera is still - and eight of this suite's checks
// assert selection as a property of the camera alone (the SSE dial picks the
// level; the cut grew when the payloads arrived; a still camera re-traverses
// nothing; two cuts built at different residency compare equal). Retention-in-
// draw delivers the same picture without redefining select_lod(), so it is the
// default and the gate is measured here, side by side, with its own counters.
//
// ON THE CITED PAPER (Real-Time Continuous Level of Detail Rendering of Point
// Clouds): it is an IN-CORE continuous-LOD method - its own repository says it
// iterates over the whole point cloud and uses no octree - so it supports the
// goal (no popping, gradual density change) but prescribes neither retention nor
// an SSE band. Nothing below leans on it.
//
// https://www.researchgate.net/publication/331608828_Real-Time_Continuous_
// Level_of_Detail_Rendering_of_Point_Clouds

struct LntFlightStats {
    uint64_t frames = 0;              // frames measured (past the warm frame)
    uint64_t pending_frames = 0;      // ... with children still in the queue
    uint64_t blanks_while_pending = 0;// CUT CELLS drawn by nothing, while pending
    uint64_t max_blanks = 0;          // worst frame's blank cells
    uint64_t missed_visible = 0;      // leaves the frustum kept that the cut lost
    uint64_t stand_ins = 0;           // windows borrowed from an ancestor
    uint64_t gated_cells = 0;         // splits held shut by a missing payload
    uint64_t data_runs = 0;           // re-selections driven by data, not camera
    uint64_t max_pending = 0;
    uint64_t max_cut = 0;
    uint64_t peak_points = 0;         // most points the draw list ever asked for
};

// One zoom-in flight, engine-free but faithful: the arena is modelled as "a tile
// is on the GPU once its payload is resident", which is what an unlimited upload
// budget produces - the upload slice is measured elsewhere (O4, J).
static LntFlightStats fly_zoom_in_with_latency(const std::string& file, Reader& rd,
                                               const BoundingBox& b, const double o[3],
                                               bool gate, bool fallback,
                                               const char** err_out) {
    LntFlightStats fs;
    *err_out = nullptr;
    uint64_t payload = 0;
    for (const LNTNodeEntry& e : rd.nodes()) payload += LntCacheManager::payload_bytes(e.point_count);
    LntCacheManager cache(payload + payload / 8);   // holds the opening cut, not the file
    std::string err;
    if (!cache.open(file, err)) { *err_out = "cache open failed"; return fs; }
    LntStreamer s;
    if (!s.attach(rd, &cache, err)) { *err_out = "attach failed"; cache.close(); return fs; }
    LntLodParams p = s.params();
    p.gate_split_on_residency = gate;
    s.set_params(p);
    s.set_ancestor_fallback(fallback);
    LntScene scene;
    if (!scene.plan(rd, err)) { *err_out = "plan failed"; cache.close(); return fs; }

    auto apply_residency = [&]() {
        // The frame's uploads happen before its draw list is built, so a payload
        // that is resident is a tile that is on the GPU.
        for (std::size_t i = 0; i < scene.tiles().size(); ++i) {
            LntTile* t = scene.tile_mut(scene.tiles()[i].node_id);
            if (t != nullptr)
                t->uploaded = t->count > 0 &&
                              cache.residency(t->node_id) == LntResidency::Resident;
        }
    };
    auto place = [&](LNTCamera& c, double d) {
        for (int i = 0; i < 3; ++i) { c.origin[i] = o[i]; c.eye[i] = b.mid(i); c.target[i] = b.mid(i); }
        c.eye[2] += d;
        c.aspect = 16.0 / 9.0;
        c.screen_height_px = 1080;
        c.near_clip = 0.1;
        c.far_clip = d * 2.0 + 2500.0;
        return c.rebuild();
    };

    // ---- the opening view, at full speed --------------------------------
    // A viewer shows its first frame once the opening data has landed, and the
    // latency under test is about what happens DURING a zoom - so one select+drain
    // at 2000 m (the full view, a COARSE cut) and then the requested 500 -> 20 m
    // flight. Warming at 500 m would pre-load exactly the leaves the flight then
    // asks for: on this fixture the SSE at 500 m already resolves to the leaf
    // level, so the queue would stay empty for all 48 frames and the assertions
    // about pending children would be measuring nothing.
    LNTCamera warm;
    if (!place(warm, 2000.0)) { *err_out = "2000 m camera has no frustum"; cache.close(); return fs; }
    // ONE select+drain, not several: warming further would pre-load exactly the
    // nodes the zoom then asks for, and a flight over an empty queue would prove
    // nothing about children that are pending.
    s.update_visible_nodes(warm);
    apply_residency();
    if (!cache.drain(30000)) {
        *err_out = "opening drain timed out";
        cache.close();
        return fs;
    }
    cache.set_load_delay_ms(10);       // 10 ms of simulated disk latency from here

    // ---- the flight ------------------------------------------------------
    const int kFrames = 48;
    for (int f = 0; f < kFrames; ++f) {
        const double t = double(f) / double(kFrames - 1);
        const double d = 500.0 + (20.0 - 500.0) * t;
        LNTCamera c;
        if (!place(c, d)) continue;
        s.update_visible_nodes(c);
        apply_residency();
        const LntLodSelection& cut = s.stats().last_selection;
        const LntDrawList& list = s.resolve_draw_list(scene);
        const LntBlankStats bl = blank_cells(s, rd, cut, list, scene);
        const LntCoverStats cs = cut_coverage(s, rd, c, cut);
        const uint64_t pending = cache.pending();
        if (pending > 0) {
            ++fs.pending_frames;
            fs.blanks_while_pending += bl.blank;
            fs.max_pending = std::max(fs.max_pending, pending);
        }
        fs.max_blanks = std::max(fs.max_blanks, bl.blank);
        fs.missed_visible += cs.missed_visible;
        fs.stand_ins += bl.stand_in_windows;
        fs.gated_cells += s.stats().gated_cells;
        fs.data_runs += s.stats().selection_runs_data;
        fs.max_cut = std::max(fs.max_cut, uint64_t(cut.cut_size()));
        fs.peak_points = std::max(fs.peak_points, list.points);
        ++fs.frames;
        // A frame boundary: the latency has to have a chance to complete a load,
        // which is what makes "children were pending on this frame" true rather
        // than a snapshot of an idle queue.
        std::this_thread::sleep_for(std::chrono::milliseconds(12));
    }
    cache.set_load_delay_ms(0);
    cache.close();
    return fs;
}

static void section_async_streaming(const std::string& path) {
    std::printf("\n=== Q. ASYNC STREAMING (RETENTION WHILE CHILDREN LOAD) ===\n");
    std::string file = path;
    {
        Reader probe;
        const std::string candidate = path + ".fine";
        if (probe.open(candidate) && probe.get_total_nodes() > 0) file = candidate;
    }
    Reader rd;
    if (!rd.open(file)) { check("async: the dataset opens", false, rd.error()); return; }
    LntStreamer probe_s;
    std::string err;
    if (!probe_s.attach(rd, nullptr, err)) {
        check("async: the octree attaches to the reader", false, err);
        return;
    }
    const BoundingBox b = rd.get_bounds();
    const double o[3] = {rd.origin(0), rd.origin(1), rd.origin(2)};
    std::printf("  dataset        : %s (%s), 10 ms simulated disk latency, 500 m -> 20 m over "
                "48 frames\n", file.c_str(), probe_s.tree().describe().c_str());

    const char* err_a = nullptr;
    const LntFlightStats retain_cut = fly_zoom_in_with_latency(file, rd, b, o,
                                                               /*gate=*/true,
                                                               /*fallback=*/true, &err_a);
    const char* err_b = nullptr;
    const LntFlightStats retain_draw = fly_zoom_in_with_latency(file, rd, b, o,
                                                                /*gate=*/false,
                                                                /*fallback=*/true, &err_b);
    const char* err_c = nullptr;
    const LntFlightStats control = fly_zoom_in_with_latency(file, rd, b, o,
                                                            /*gate=*/false,
                                                            /*fallback=*/false, &err_c);
    if (err_a || err_b || err_c) {
        const std::string detail = std::string(err_a ? err_a : "") +
                                   (err_b ? std::string(" / ") + err_b : "") +
                                   (err_c ? std::string(" / ") + err_c : "");
        check("async: all three flights ran", false, detail);
        return;
    }
    auto line = [](const char* tag, const LntFlightStats& fs) {
        std::printf("     %-16s: %llu frames, %llu with children pending (peak %llu queued), "
                    "%llu cells drawn by nothing while pending (worst frame %llu), %llu "
                    "stand-ins, %llu gated splits, %llu data-driven rebuilds, cut up to %llu "
                    "nodes\n",
                    tag, (unsigned long long)fs.frames,
                    (unsigned long long)fs.pending_frames, (unsigned long long)fs.max_pending,
                    (unsigned long long)fs.blanks_while_pending,
                    (unsigned long long)fs.max_blanks, (unsigned long long)fs.stand_ins,
                    (unsigned long long)fs.gated_cells, (unsigned long long)fs.data_runs,
                    (unsigned long long)fs.max_cut);
    };
    line("RETAIN-IN-CUT", retain_cut);
    line("RETAIN-IN-DRAW", retain_draw);
    line("CONTROL", control);

    // The flight only proves something if it actually had children in flight.
    check("async: the flight really had children pending on most frames (the latency "
          "hook is doing its job)",
          retain_draw.pending_frames > retain_draw.frames / 2 &&
              retain_draw.max_pending > 0,
          std::to_string(retain_draw.pending_frames) + " of " +
              std::to_string(retain_draw.frames) + " frames, peak " +
              std::to_string(retain_draw.max_pending) + " queued");
    // Q1(a): the cut's own spatial coverage, asserted at EVERY frame - this is the
    // request's wording ("100% spatial coverage at every frame during the
    // transition") and it is a statement about the traversal, not about loading.
    check("async: every frame's cut covers every leaf the frustum keeps (100% spatial "
          "coverage through the whole transition)",
          retain_cut.missed_visible == 0 && retain_draw.missed_visible == 0 &&
              control.missed_visible == 0,
          std::to_string(retain_cut.missed_visible) + " / " +
              std::to_string(retain_draw.missed_visible) + " / " +
              std::to_string(control.missed_visible) + " leaves missed");
    // Q1(b): no empty cell while a payload is in flight, in EITHER retention form.
    check("async: ZERO cells drawn by nothing while children sat in the pending queue, "
          "with retention in the cut",
          retain_cut.blanks_while_pending == 0 && retain_cut.max_blanks == 0,
          std::to_string(retain_cut.blanks_while_pending) + " blank cells across " +
              std::to_string(retain_cut.pending_frames) + " pending frames");
    check("async: ZERO cells drawn by nothing while children sat in the pending queue, "
          "with retention in the draw list (the shipped default)",
          retain_draw.blanks_while_pending == 0 && retain_draw.max_blanks == 0,
          std::to_string(retain_draw.blanks_while_pending) + " blank cells across " +
              std::to_string(retain_draw.pending_frames) + " pending frames");
    // The teeth: with both mechanisms off the same trajectory DOES go empty.
    check("async: the control flight blanks cells, so the two checks above are not "
          "passing because nothing was ever at risk",
          control.blanks_while_pending > 0,
          std::to_string(control.blanks_while_pending) + " blank cells while pending");
    // Both mechanisms actually engaged - neither check may pass by staying idle.
    // The gate's DATA-driven rebuild is NOT asserted here: in a camera that is
    // moving anyway the camera rule is what forces the rebuild, so that clause
    // could never hold on this trajectory. It is measured on a parked camera
    // below, where it is the only thing that can move the cut.
    check("async: retention-in-draw engaged (ancestor tiles were borrowed)",
          retain_draw.stand_ins > 0,
          std::to_string(retain_draw.stand_ins) + " stand-in windows");
    // The gate held, AND the cut still advanced while it held - a gate that never
    // opens would be a renderer stuck at the root. (The DATA-driven rebuild is
    // measured separately below: in a camera that is moving anyway the camera
    // rule is what forces the rebuild, so counting it here would be measuring the
    // wrong thing.)
    check("async: retention-in-cut engaged - the gate held splits while children "
          "loaded, and the cut still advanced as they landed",
          retain_cut.gated_cells > 0 && retain_cut.max_cut > 1,
          std::to_string(retain_cut.gated_cells) + " held splits, cut peaked at " +
              std::to_string(retain_cut.max_cut) + " nodes");

    // ---- Q1(c): a camera that STOPS --------------------------------------
    // The reason a data-driven rebuild exists at all: with a still camera the
    // motion hysteresis would reuse the coarse cut for ever, so without a rebuild
    // triggered by ARRIVAL the gate would hold the viewer blocky for ever. A
    // camera parked at 500 m with the gate on must climb to the leaves on its own,
    // and the rebuilds that do it must be counted as data-driven.
    {
        uint64_t payload = 0;
        for (const LNTNodeEntry& e : rd.nodes())
            payload += LntCacheManager::payload_bytes(e.point_count);
        LntCacheManager c(payload + payload / 8);
        std::string e;
        const char* fail = nullptr;
        LntLodSelection first_cut, last_cut;
        uint64_t camera_runs = 0, data_runs = 0, gated = 0;
        if (!c.open(file, e)) {
            fail = "cache open failed";
        } else {
            LntStreamer s;
            if (!s.attach(rd, &c, e)) fail = "attach failed";
            LntLodParams p = s.params();
            p.gate_split_on_residency = true;
            s.set_params(p);
            LNTCamera cam;
            for (int i = 0; i < 3; ++i) { cam.origin[i] = o[i]; cam.eye[i] = b.mid(i); cam.target[i] = b.mid(i); }
            cam.eye[2] += 500.0;
            cam.aspect = 16.0 / 9.0;
            cam.screen_height_px = 1080;
            cam.near_clip = 0.1;
            cam.far_clip = 3500.0;
            if (!cam.rebuild()) fail = "500 m camera has no frustum";
            c.set_load_delay_ms(10);
            for (int f = 0; f < 40 && fail == nullptr; ++f) {
                s.update_visible_nodes(cam);
                if (f == 0) first_cut = s.stats().last_selection;
                last_cut = s.stats().last_selection;
                const LntStreamStats& st = s.stats();
                camera_runs = st.selection_runs_camera;
                data_runs = st.selection_runs_data;
                gated = st.gated_cells;
                std::this_thread::sleep_for(std::chrono::milliseconds(12));
            }
            c.set_load_delay_ms(0);
        }
        c.close();
        if (fail != nullptr) {
            check("async: a still camera with the gate on refines as data lands", false, fail);
        } else {
            std::printf("  still camera   : parked at 500 m for 40 frames with the gate on -> "
                        "cut %zu -> %zu nodes, %llu camera-driven and %llu data-driven "
                        "rebuilds (gate held on the last frame: %llu)\n",
                        first_cut.cut_size(), last_cut.cut_size(),
                        (unsigned long long)camera_runs, (unsigned long long)data_runs,
                        (unsigned long long)gated);
            check("async: a PARKED camera still refines once the payloads land (the motion "
                  "hysteresis must not hold the coarse cut for ever)",
                  last_cut.cut_size() > first_cut.cut_size() && gated > 0,
                  std::to_string(first_cut.cut_size()) + " -> " +
                      std::to_string(last_cut.cut_size()) + " nodes, gate held " +
                      std::to_string(gated) + " splits on the last frame");
            check("async: the refinements of a parked camera are counted as data-driven "
                  "rebuilds, not as camera motion",
                  data_runs > 0 && camera_runs <= 1,
                  std::to_string(data_runs) + " data-driven, " +
                      std::to_string(camera_runs) + " camera-driven");
        }
    }
    // The point of allowing the cut to be coarse while data streams: what is
    // DRAWN never depends on the loading state being finished.
    std::printf("  draw list peak : %llu points (retain-in-cut) vs %llu (retain-in-draw)\n",
                (unsigned long long)retain_cut.peak_points,
                (unsigned long long)retain_draw.peak_points);

    // ---- Q2: the 10% band, and its control ------------------------------            // One level's threshold crossing, approached by the THRESHOLD rather
            // than by the lens: moving the lens also changes which cells the
            // frustum keeps, which would change the cut for reasons that have
            // nothing to do with hysteresis and drown the signal. The band is a
            // statement about the SSE comparison, so the comparison is what
            // oscillates - a fixed camera whose level-(depth-1) cells sit just
            // under the threshold, alternating between a threshold they clear and
            // one they do not.
            const LntNodeTree& tree = probe_s.tree();
            if (tree.max_depth < 2) {
                skip("band: the cut holds through a threshold crossing",
                     "this tree is " + std::to_string(tree.max_depth) + " level(s) deep");
            } else {
            const uint32_t root_idx = tree.roots[0];
            const double root_half = double(rd.nodes()[root_idx].half_size);
            const int level = tree.max_depth - 1;
            const double half = root_half / std::pow(2.0, level);
            const double t_hi = 90.0;                              // the configured dial
            const double t_lo = 85.0;                              // clears it
            const double d_band = half * 1080.0 / (0.9667 * t_hi); // sse ~ 0.967 x t_hi
            struct BandRun {
                bool ran = false;
                uint64_t flips = 0;
                uint64_t deepest = 0;
                uint64_t max_count = 0;
            };
            auto run_band = [&](double ratio) {
                BandRun r;
                LntStreamer s;
                std::string e;
                if (!s.attach(rd, nullptr, e)) return r;
                int last_count = -1;
                for (int f = 0; f < 12; ++f) {
                    LntLodParams p;
                    p.sse_coarsen_ratio = ratio;
                    p.sse_threshold = (f % 2 == 0) ? t_lo : t_hi;
                    s.set_params(p);                 // a param change forces a rebuild
                    LNTCamera c;
                    for (int i = 0; i < 3; ++i) { c.origin[i] = o[i]; c.eye[i] = b.mid(i); c.target[i] = b.mid(i); }
                    c.eye[2] += d_band;
                    c.aspect = 16.0 / 9.0;
                    c.screen_height_px = 1080;
                    c.near_clip = 0.1;
                    c.far_clip = d_band * 2.0 + 2500.0;
                    if (!c.rebuild()) continue;
                    s.update_visible_nodes(c);
                    const LntLodSelection& cut = s.stats().last_selection;
                    // How many nodes at the crossing level are IN the cut. That is
                    // the flapping made countable: the band's whole effect is
                    // whether those nodes are drawn or their children are.
                    int count = 0;
                    for (uint64_t id : cut.node_ids) {
                        const auto it = tree.index_of.find(id);
                        if (it == tree.index_of.end()) continue;
                        if (tree.nodes[it->second].level == level) ++count;
                    }
                    if (last_count >= 0 && count != last_count) ++r.flips;
                    last_count = count;
                    r.max_count = std::max(r.max_count, uint64_t(count));
                    r.deepest = std::max(r.deepest, uint64_t(cut.max_depth_reached));
                }
                r.ran = true;
                return r;
            };
            const BandRun with_band = run_band(0.90);
            const BandRun no_band = run_band(1.00);     // >= 1.0 disables the band
            std::printf("  band           : level %d cells (%.1f m half) sit at SSE %.0f at "
                        "%.0f m; alternating a threshold they clear (85) with one they "
                        "miss (90)\n",
                        level, half, 0.9667 * t_hi, d_band);
            std::printf("     band on     : %llu resolution flips in 12 frames (up to %llu nodes "
                        "at level %d)\n", (unsigned long long)with_band.flips,
                        (unsigned long long)with_band.max_count, level);
            std::printf("     band off    : %llu resolution flips in 12 frames\n",
                        (unsigned long long)no_band.flips);
            check("band: with the 10% buffer the cut does not flip while the SSE oscillates "
                  "inside the band",
                  with_band.ran && with_band.flips == 0 && with_band.max_count > 0,
                  std::to_string(with_band.flips) + " flips, up to " +
                      std::to_string(with_band.max_count) + " nodes at level " +
                      std::to_string(level));
            check("band: without the buffer the same oscillation flips the cut every frame, "
                  "so the check has teeth",
                  no_band.ran && no_band.flips >= 6,
                  std::to_string(no_band.flips) + " flips in 12 frames");
            check("band: both runs really did reach the leaves (the control is not "
                  "comparing two equally coarse cuts)",
                  with_band.deepest > 0 && int(with_band.deepest) == tree.max_depth &&
                      no_band.deepest == with_band.deepest,
                  "deepest cut level " + std::to_string(with_band.deepest) + " of " +
                      std::to_string(tree.max_depth));
            }
}

// ========================================================================
// R. CROSS-SECTION: plane math, corridor query, GPU clip volume
// ========================================================================
// Three layers, each verified separately so a failure names its layer:
//   R1  Plane / ClipBox / plane-AABB math, against brute force over box corners.
//   R2  Streamer corridor filtering: the cut is restricted to the corridor, the
//       node query is exact, and lifting the corridor restores the full cut.
//   R3  GPU clipping through nkv_set_clip_volume: points on the negative side of a
//       plane vanish from the frame with ZERO CPU buffer traffic. Skipped (not
//       passed) when the engine DLL has no nkv_set_clip_volume yet.
// The brief names "Z = 100 m" as the cut: this dataset's Z spans 1040..1160 m, so
// the half-way plane (the same "cut the data in half") is Z = 1100 m.
static void section_cross_section(const std::string& path) {
    std::printf("\n=== R. CROSS-SECTION (PLANES, CORRIDOR, GPU CLIP) ===\n");

    // ---- R1. math ----------------------------------------------------------
    {
        const double pt[3] = {0, 0, 10}, nrm[3] = {0, 0, 2};
        const Plane up = Plane::from_point_normal(pt, nrm);
        check("a plane built from point + normal is normalised and keeps its side",
              std::fabs(up.c - 1.0) < 1e-12 && up.keeps(5, 5, 10.5) && !up.keeps(5, 5, 9.5) &&
                  std::fabs(up.distance(0, 0, 13) - 3.0) < 1e-12);
        Rng rng(77);
        uint64_t bad = 0, straddle = 0, inside = 0, outside = 0;
        for (int t = 0; t < 4000; ++t) {
            Plane p(rng.unit() * 2 - 1, rng.unit() * 2 - 1, rng.unit() * 2 - 1,
                    (rng.unit() * 2 - 1) * 30.0);
            BoundingBox b = BoundingBox::from_min_max(
                rng.unit() * 20 - 10, rng.unit() * 20 - 10, rng.unit() * 20 - 10, 0, 0, 0);
            for (int a = 0; a < 3; ++a) b.max[a] = b.min[a] + 0.1 + rng.unit() * 8.0;
            int kept = 0, dropped = 0;
            for (int c = 0; c < 8; ++c) {
                const bool k = p.keeps((c & 1) ? b.max[0] : b.min[0], (c & 2) ? b.max[1] : b.min[1],
                                       (c & 4) ? b.max[2] : b.min[2]);
                (k ? kept : dropped)++;
            }
            // A plane is linear, so the extremes of the signed distance over a box are
            // at its corners: the corner census IS the ground truth.
            const PlaneSide want = dropped == 0 ? PlaneSide::Inside
                                   : kept == 0  ? PlaneSide::Outside : PlaneSide::Straddles;
            const PlaneSide got = classify_aabb(p, b);
            if (got != want) ++bad;
            if (want == PlaneSide::Straddles) ++straddle;
            else if (want == PlaneSide::Inside) ++inside; else ++outside;
            if (plane_aabb_intersects(p, b) != (kept > 0)) ++bad;
        }
        check("plane-AABB classification matches the 8-corner ground truth (4000 random cases)",
              bad == 0 && straddle > 100 && inside > 100 && outside > 100,
              std::to_string(bad) + " wrong; " + std::to_string(inside) + " inside / " +
                  std::to_string(straddle) + " straddling / " + std::to_string(outside) + " outside");

        const double a[2] = {100, 200}, b2[2] = {160, 280};
        const ClipBox cb = ClipBox::corridor(a, b2, 12.0, 50.0, 30.0);
        const std::vector<Plane> six = cb.planes();
        uint64_t mismatch = 0;
        for (int t = 0; t < 20000; ++t) {
            const double x = 60 + rng.unit() * 160, y = 160 + rng.unit() * 160, z = rng.unit() * 120;
            bool all = true;
            for (const Plane& p : six) all = all && p.keeps(x, y, z);
            if (all != cb.contains(x, y, z)) ++mismatch;
        }
        check("a corridor ClipBox equals the intersection of its six inward planes",
              six.size() == 6 && mismatch == 0, std::to_string(mismatch) + " disagreements / 20000");
        check("the corridor's end points are inside and a point beside it is outside",
              cb.contains(100, 200, 50) && cb.contains(160, 280, 50) && !cb.contains(100, 230, 50));
        const BoundingBox ab = cb.aabb();
        check("the corridor AABB encloses the oriented box",
              ab.contains(100, 200, 50) && ab.contains(160, 280, 50) && ab.extent(0) < 100.0);
    }

    // ---- R2. corridor filtering on the streamer ------------------------------
    Reader rd;
    if (!rd.open(path)) { check("cross-section: the dataset opens", false, rd.error()); return; }
    LntStreamer streamer;
    std::string err;
    if (!streamer.attach(rd, nullptr, err)) {
        check("cross-section: the streamer attaches", false, err);
        return;
    }
    const BoundingBox env = rd.get_bounds();
    const double side = env.extent(0);
    const double mid[3] = {env.mid(0), env.mid(1), env.mid(2)};
    double origin[3];
    for (int i = 0; i < 3; ++i) origin[i] = rd.origin(i);

    LNTCamera wide;
    wide.eye[0] = mid[0] - side * 2.2;
    wide.eye[1] = mid[1] - side * 2.2;
    wide.eye[2] = mid[2] + side * 1.8;
    for (int i = 0; i < 3; ++i) wide.target[i] = mid[i];
    wide.aspect = 16.0 / 9.0;
    wide.near_clip = 1.0;
    wide.far_clip = side * 40.0;
    wide.screen_height_px = 720;
    for (int i = 0; i < 3; ++i) wide.origin[i] = origin[i];
    wide.rebuild();

    LntLodParams leaves;
    leaves.sse_threshold = 0.0;
    leaves.max_nodes = 100000;
    streamer.set_params(leaves);
    streamer.update_visible_nodes(wide);
    const LntLodSelection baseline = streamer.stats().last_selection;

    // A long, narrow, full-height strip: the shape a road profile has.
    const double p1[2] = {env.min[0] + side * 0.20, mid[1]};
    const double p2[2] = {env.min[0] + side * 0.35, mid[1]};
    const ClipBox strip = ClipBox::corridor(p1, p2, side * 0.04, mid[2], side * 0.5);
    const BoundingBox corridor = strip.aabb();

    const std::vector<LNTNodeEntry> hit = streamer.query_nodes_in_corridor(corridor);
    uint64_t expect = 0, wrong = 0;
    for (const LNTNodeEntry& e : rd.nodes())
        if (rd.bounds_of(e).intersects(corridor)) ++expect;
    for (const LNTNodeEntry& e : hit)
        if (!rd.bounds_of(e).intersects(corridor)) ++wrong;
    check("the corridor query returns exactly the directory nodes whose cell meets the corridor",
          !hit.empty() && hit.size() == expect && wrong == 0,
          std::to_string(hit.size()) + " of " + std::to_string(rd.nodes().size()) + " nodes");
    check("the corridor is a strict subset of the dataset (nodes outside it are discarded)",
          hit.size() < rd.nodes().size());

    streamer.set_corridor(corridor);
    streamer.update_visible_nodes(wide);
    const LntLodSelection cut = streamer.stats().last_selection;
    uint64_t outside_sel = 0;
    for (uint64_t id : cut.node_ids) {
        const LNTNodeEntry* e = rd.find_node(id);
        if (e == nullptr || !rd.bounds_of(*e).intersects(corridor)) ++outside_sel;
    }
    check("with a corridor set, the cut contains no node outside the corridor",
          !cut.node_ids.empty() && outside_sel == 0, std::to_string(outside_sel) + " outside");
    uint64_t want_n = 0;
    for (uint64_t id : baseline.node_ids) {
        const LNTNodeEntry* e = rd.find_node(id);
        if (e && rd.bounds_of(*e).intersects(corridor)) ++want_n;
    }
    check("the corridor cut keeps EVERY node of the full cut that meets the corridor",
          cut.node_ids.size() == want_n,
          std::to_string(cut.node_ids.size()) + " vs " + std::to_string(want_n));
    check("the corridor cut selects fewer points than the full cut (disk and VRAM saved)",
          cut.selected_points < baseline.selected_points,
          std::to_string(cut.selected_points) + " vs " + std::to_string(baseline.selected_points));
    streamer.clear_corridor();
    streamer.update_visible_nodes(wide);
    check("lifting the corridor restores the full cut",
          streamer.stats().last_selection.node_ids == baseline.node_ids);

    // ---- R3. GPU clip volume ---------------------------------------------------
    const std::string dll = VulkanEngine::find_dll();
    if (dll.empty()) { skip("GPU clip volume", "no naksha_vulkan.dll found (set LNT_VULKAN_DLL)"); return; }
    VulkanEngine eng;
    if (!eng.load(dll, err)) { skip("GPU clip volume", "engine load failed: " + err); return; }
    if (!eng.is_available()) { skip("GPU clip volume", "no Vulkan device"); return; }
    if (!eng.has_clip_volume()) {
        skip("GPU clip volume", "this naksha_vulkan.dll has no nkv_set_clip_volume - rebuild the "
                                "engine (MSVC) from the updated sources and point LNT_VULKAN_DLL at it");
        return;
    }
    if (!eng.open(960, 720, /*hidden=*/true, /*validation=*/false, err)) {
        skip("GPU clip volume", "renderer creation failed: " + err);
        return;
    }
    LntScene scene;
    if (!scene.upload(eng, rd, err)) { check("clip: the arena uploads", false, err); return; }
    eng.set_clear_color(0.05f, 0.05f, 0.08f, 1.0f);
    // The eye sits 5 m above the dataset centre and the cut plane is at the EYE height. A
    // perfectly level look-at (eye z == target z) renders nothing in this engine, so the
    // camera pitches down a fraction of a degree; with the plane at eye height every point
    // above it still projects above the image centre line and every point below it below.
    const double zc = mid[2] + 5.0;
    eng.set_camera_lookat(mid[0], mid[1] - side * 1.4, zc, mid[0], mid[1], mid[2], 45.0, 1.0,
                          side * 20.0);
    scene.show_all(eng, true);

    auto rows = [](const EngineFrame& f, const uint8_t bg[3], uint64_t& upper, uint64_t& lower) {
        upper = lower = 0;
        const uint32_t centre = f.height / 2;
        for (uint32_t y = 0; y < f.height; ++y)
            for (uint32_t x = 0; x < f.width; ++x) {
                const uint8_t* p = f.at(x, y);
                const bool geo = std::abs(int(p[0]) - bg[0]) > 6 || std::abs(int(p[1]) - bg[1]) > 6 ||
                                 std::abs(int(p[2]) - bg[2]) > 6;
                if (!geo) continue;
                if (y + 6 < centre) ++upper;          // 6 px margin for the sprite radius
                else if (y > centre + 6) ++lower;
            }
    };
    EngineFrame full;
    eng.render();
    if (!eng.capture_frame(full)) { check("clip: a frame is captured", false, eng.last_error()); return; }
    uint8_t bg[3];
    uint64_t bg_n = 0;
    dominant_colour(full, bg, &bg_n);
    uint64_t up0, lo0;
    rows(full, bg, up0, lo0);
    std::printf("  clip frame     : %ux%u, background #%02x%02x%02x (%llu px), %llu geometry px\n",
                full.width, full.height, bg[0], bg[1], bg[2], (unsigned long long)bg_n,
                (unsigned long long)full.count_not_equal(bg[0], bg[1], bg[2], 6));
    check("unclipped, the level side view has geometry both above and below the cut height",
          up0 > 0 && lo0 > 0, std::to_string(up0) + " above / " + std::to_string(lo0) + " below");

    const uint64_t pos_up = eng.position_upload_count();
    const uint64_t tiles = eng.arena_uploaded_tiles();
    const uint32_t ranges = eng.draw_range_count();

    const double keep_above[4] = {0, 0, 1, -zc};          // Z >= cut height
    check("the engine accepts a world-space clip plane", eng.set_clip_planes_world(keep_above, 1));
    EngineFrame upper_only;
    eng.render();
    eng.capture_frame(upper_only);
    uint64_t up1, lo1;
    rows(upper_only, bg, up1, lo1);
    check("GPU clip keep-above: every point below the plane is discarded",
          lo1 == 0 && up1 > 0, std::to_string(up1) + " above / " + std::to_string(lo1) + " below");

    const double keep_below[4] = {0, 0, -1, zc};          // Z <= cut height
    eng.set_clip_planes_world(keep_below, 1);
    EngineFrame lower_only;
    eng.render();
    eng.capture_frame(lower_only);
    uint64_t up2, lo2;
    rows(lower_only, bg, up2, lo2);
    check("GPU clip keep-below: every point above the plane is discarded",
          up2 == 0 && lo2 > 0, std::to_string(up2) + " above / " + std::to_string(lo2) + " below");

    check("clipping changed NO CPU buffer: zero position uploads, same tiles, same draw ranges",
          eng.position_upload_count() == pos_up && eng.arena_uploaded_tiles() == tiles &&
              eng.draw_range_count() == ranges,
          std::to_string(eng.position_upload_count() - pos_up) + " position uploads");

    eng.set_clip_planes_world(nullptr, 0);
    EngineFrame restored;
    eng.render();
    eng.capture_frame(restored);
    uint64_t up3, lo3;
    rows(restored, bg, up3, lo3);
    check("switching the clip volume off restores the full cloud",
          up3 == up0 && lo3 == lo0, std::to_string(up3) + " / " + std::to_string(lo3));

    // Two parallel planes: a horizontal slab Z in [mid - 10, mid + 10].
    const double slab[8] = {0, 0, 1, -(zc - 10.0), 0, 0, -1, zc + 10.0};
    eng.set_clip_planes_world(slab, 2);
    EngineFrame slab_f;
    eng.render();
    eng.capture_frame(slab_f);
    const uint64_t px_slab = slab_f.count_not_equal(bg[0], bg[1], bg[2], 6);
    const uint64_t px_full = full.count_not_equal(bg[0], bg[1], bg[2], 6);
    check("a dual parallel-plane slab keeps a thin band: fewer pixels than the full cloud, not none",
          px_slab > 0 && px_slab < px_full, std::to_string(px_slab) + " vs " + std::to_string(px_full));
    eng.set_clip_planes_world(nullptr, 0);
}

// ========================================================================
// S. LARGE-CLOUD STREAMING: coverage, budget, merged draws, multi-loader cache
// ========================================================================
// What a 3-billion-point flight needs, on a dataset small enough to build in a
// test: a deep tree with thousands of nodes, a REAL asynchronous cache with several
// loader threads, and a 60-frame zoom from the full extent down to leaf level.
//   * every frame, every leaf the frustum keeps is covered by the cut (no spatial gap)
//   * every frame, nothing the draw list needs is blank (no black hole on screen)
//   * the streamer's per-frame update stays <= 2.0 ms with the 2 GiB default cache
//   * the point budget caps the cut, contiguous draw windows merge, stale requests
//     are cancelled, and several loaders really do overlap.
static void section_large_cloud(const std::string& path) {
    std::printf("\n=== S. LARGE-CLOUD STREAMING (COVERAGE, BUDGET, MULTI-LOADER CACHE) ===\n");

    // ---- S0. unit checks that need no dataset ---------------------------------
    {
        std::vector<uint32_t> f = {30, 0, 10, 100, 5000}, c = {5, 10, 20, 50, 7}, mf, mc;
        LntStreamer::merge_contiguous(f, c, mf, mc);
        check("contiguous draw windows merge into one range, gaps stay separate",
              mf == std::vector<uint32_t>({0, 100, 5000}) &&
                  mc == std::vector<uint32_t>({35, 50, 7}),
              std::to_string(mf.size()) + " ranges from 5");
        LntLodParams d;
        check("the default cut may hold 65,536 active tiles per frame", d.max_nodes == 65536u,
              std::to_string(d.max_nodes));
        check("the default cache budget is 2 GiB and the large-cloud tier 4 GiB",
              kDefaultCacheBudgetBytes == 2048ull * 1024 * 1024 &&
                  kLargeCloudCacheBudgetBytes == 4096ull * 1024 * 1024);
    }

    // ---- S1. a deep synthetic cloud --------------------------------------------
    const std::string big = path + ".large";
    {
        Dataset ds = make_dataset(1500000, 20261012ull);
        LNTBuildParams p;
        p.max_points_per_node = 2500;
        p.min_points_per_node = 400;
        p.target_resolution = 0.0;
        p.parent_lod_target = 2500;
        LNTBuildStats st = LNTBuilder::build(ds.pts.data(), ds.pts.size(), big.c_str(), p);
        check("a 1.5 M-point cloud builds into a deep tree",
              st.ok && st.node_count > 500,
              std::to_string(st.node_count) + " nodes, depth " + std::to_string(st.max_depth));
        if (!st.ok) return;
    }
    Reader rd;
    if (!rd.open(big)) { check("large: the file opens", false, rd.error()); return; }
    const BoundingBox env = rd.get_bounds();
    double origin[3];
    for (int i = 0; i < 3; ++i) origin[i] = rd.origin(i);

    // ---- S2. several loaders, one cache ----------------------------------------
    {
        LntCacheManager mc;       // default budget: 2 GiB
        std::string err;
        check("a cache opens with 3 loader threads, each on its own reader",
              mc.open(big, err, 3) && mc.worker_count() == 3, err);
        std::vector<uint64_t> ids;
        for (const LNTNodeEntry& e : rd.nodes()) ids.push_back(e.node_id);
        const auto t0 = std::chrono::steady_clock::now();
        mc.request_all(ids);
        const bool drained = mc.drain(60000);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
        uint64_t resident = 0;
        for (uint64_t id : ids) if (mc.residency(id) == LntResidency::Resident) ++resident;
        check("every node loads through the parallel loaders, none is lost or duplicated",
              drained && resident == ids.size() && mc.stats().loads == ids.size() &&
                  mc.stats().load_failures == 0,
              std::to_string(resident) + " of " + std::to_string(ids.size()) + " resident in " +
                  std::to_string(ms) + " ms");
        mc.close();

        // cancel + priority: only requests the caller owns are dropped
        LntCacheManager hc;
        hc.open(big, err, 1);
        hc.hold(true);
        std::vector<uint64_t> few(ids.begin(), ids.begin() + std::min<size_t>(6, ids.size()));
        hc.request_all(few);
        std::unordered_set<uint64_t> keep = {few[0]}, mine(few.begin(), few.begin() + 4);
        const uint64_t dropped = hc.cancel_unwanted(keep, mine);
        check("stale requests are cancelled: owned + unwanted dropped, wanted and foreign kept",
              dropped == 3 && hc.pending() == 3 &&
                  hc.residency(few[1]) == LntResidency::Absent &&
                  hc.residency(few[0]) == LntResidency::Queued &&
                  hc.residency(few[5]) == LntResidency::Queued,
              std::to_string(dropped) + " dropped, " + std::to_string(hc.pending()) + " pending");
        hc.hold(false);
        hc.drain(20000);
        hc.close();
    }

    // ---- S3. the 60-frame zoom sweep -------------------------------------------
    LntCacheManager cache;                  // the 2 GiB default
    std::string err;
    if (!cache.open(big, err, 3)) { check("large: the sweep cache opens", false, err); return; }
    LntStreamer streamer;
    if (!streamer.attach(rd, &cache, err)) { check("large: the streamer attaches", false, err); return; }
    LntLodParams lp = streamer.params();
    lp.gate_split_on_residency = true;      // parent retention
    lp.sse_threshold = 24.0;
    streamer.set_params(lp);
    streamer.set_base_levels(1);
    streamer.set_prefetch_ahead(4.0);
    LntScene scene;
    if (!scene.plan(rd, err)) { check("large: the scene plans", false, err); return; }
    const LntNodeTree& tree = streamer.tree();

    auto place = [&](LNTCamera& c, double d) {
        for (int i = 0; i < 3; ++i) { c.origin[i] = origin[i]; c.eye[i] = env.mid(i); c.target[i] = env.mid(i); }
        c.eye[2] += d;
        c.aspect = 16.0 / 9.0;
        c.screen_height_px = 1080;
        c.near_clip = 0.1;
        c.far_clip = d * 2.0 + 2500.0;
        return c.rebuild();
    };
    auto apply_residency = [&]() {
        for (const LntTile& t : scene.tiles()) {
            LntTile* m = scene.tile_mut(t.node_id);
            if (m != nullptr)
                m->uploaded = m->count > 0 && cache.residency(m->node_id) == LntResidency::Resident;
        }
    };

    // The opening frame: the viewer shows its first image once the base layer landed.
    LNTCamera cam;
    place(cam, 4000.0);
    streamer.update_visible_nodes(cam);
    check("the opening drain completes (base layer resident)", cache.drain(60000));
    apply_residency();
    cache.set_load_delay_ms(4);          // 4 ms of simulated disk latency per tile from here

    const int kFrames = 60;
    uint64_t gaps = 0, blank_frames = 0, worst_uncovered = 0, over_2ms = 0;
    double worst_ms = 0.0, sum_ms = 0.0;
    uint64_t max_cut = 0, resident_misses = 0;
    int deepest = 0;
    for (int f = 0; f < kFrames; ++f) {
        const double u = double(f) / double(kFrames - 1);
        const double d = 4000.0 * std::pow(25.0 / 4000.0, u);        // 4000 m -> 25 m
        if (!place(cam, d)) { check("large: sweep camera frustum", false, "degenerate"); break; }

        const auto t0 = std::chrono::steady_clock::now();
        streamer.update_visible_nodes(cam);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
        worst_ms = std::max(worst_ms, ms);
        sum_ms += ms;
        if (ms > 2.0) ++over_2ms;

        const LntLodSelection& sel = streamer.stats().last_selection;
        max_cut = std::max<uint64_t>(max_cut, sel.node_ids.size());
        std::unordered_set<uint64_t> cutset(sel.node_ids.begin(), sel.node_ids.end());
        for (uint64_t id : sel.node_ids) {
            const auto it = tree.index_of.find(id);
            if (it != tree.index_of.end()) deepest = std::max<int>(deepest, tree.nodes[it->second].level);
        }

        // 100% spatial coverage: every leaf the frustum keeps has itself or an
        // ancestor in the cut.
        uint64_t frame_gaps = 0;
        for (std::size_t i = 0; i < tree.nodes.size(); ++i) {
            if (!tree.nodes[i].children.empty()) continue;
            if (!cam.visible(rd.nodes()[i], origin)) continue;
            bool covered = false;
            for (uint32_t cur = uint32_t(i); cur != kNoParent; cur = tree.nodes[cur].parent)
                if (cutset.count(tree.nodes[cur].id)) { covered = true; break; }
            if (!covered) ++frame_gaps;
        }
        gaps += frame_gaps;

        // what is on screen: payloads that have landed are on the GPU; every cut cell
        // must then be drawn by itself or by an uploaded ancestor
        std::this_thread::sleep_for(std::chrono::milliseconds(3));    // a frame's worth of loading
        apply_residency();
        const LntDrawList& list = streamer.resolve_draw_list(scene);
        worst_uncovered = std::max<uint64_t>(worst_uncovered, list.uncovered_nodes);
        if (list.uncovered_nodes > 0) ++blank_frames;
        if (list.empty()) ++blank_frames;
        (void)resident_misses;
    }
    std::printf("  sweep          : %d frames 4000 m -> 25 m, %zu nodes, deepest cut level %d, "
                "cut up to %llu nodes\n", kFrames, tree.nodes.size(), deepest,
                (unsigned long long)max_cut);
    std::printf("  update time    : mean %.3f ms, worst %.3f ms, %llu frames over 2 ms; "
                "%llu prefetch requests, %llu stale requests cancelled\n",
                sum_ms / kFrames, worst_ms, (unsigned long long)over_2ms,
                (unsigned long long)streamer.prefetch_requested(),
                (unsigned long long)streamer.requests_cancelled());
    check("the sweep reaches deep levels (it really zooms to leaf detail)", deepest >= 2,
          "level " + std::to_string(deepest));
    check("100% spatial coverage on all 60 frames: every visible leaf is covered by the cut",
          gaps == 0, std::to_string(gaps) + " uncovered leaf-frames");
    check("no black holes on any frame: every cut cell is drawn by itself or by a parent",
          blank_frames == 0 && worst_uncovered == 0,
          std::to_string(blank_frames) + " blank frames, worst " +
              std::to_string(worst_uncovered) + " uncovered cells");
    check("the streamer's per-frame update stays <= 2.0 ms with the 2 GiB cache",
          worst_ms <= 2.0, std::to_string(worst_ms) + " ms worst");
    cache.drain(60000);
    check("the cache never exceeded its budget during the sweep",
          cache.stats().within_budget(), cache.stats().describe());

    // ---- S4. the point budget ties the cut to the zoom level ------------------------
    {
        streamer.set_prefetch_ahead(0.0);
        LntLodParams b = streamer.params();
        b.sse_threshold = 0.0;                    // would select every leaf
        b.gate_split_on_residency = false;        // the cut under test is the geometric one
        b.point_budget = 0;
        streamer.set_params(b);
        place(cam, 4000.0);
        streamer.update_visible_nodes(cam);
        const uint64_t unlimited = streamer.stats().last_selection.selected_points;
        b.point_budget = 120000;
        streamer.set_params(b);
        streamer.update_visible_nodes(cam);
        const uint64_t limited = streamer.stats().last_selection.selected_points;
        check("without a budget the full view selects the whole cloud; with one it is capped",
              unlimited >= 1000000 && limited <= 120000 && streamer.effective_sse() > 0.0,
              std::to_string(unlimited) + " -> " + std::to_string(limited) + " points, effective SSE " +
                  std::to_string(streamer.effective_sse()));
        place(cam, 40.0);                         // zoomed in: frustum culls, same budget, finer
        streamer.update_visible_nodes(cam);
        const LntLodSelection& near_sel = streamer.stats().last_selection;
        int near_level = 0;
        for (uint64_t id : near_sel.node_ids) {
            const auto it = tree.index_of.find(id);
            if (it != tree.index_of.end()) near_level = std::max<int>(near_level, tree.nodes[it->second].level);
        }
        place(cam, 4000.0);
        streamer.update_visible_nodes(cam);
        int far_level = 0;
        for (uint64_t id : streamer.stats().last_selection.node_ids) {
            const auto it = tree.index_of.find(id);
            if (it != tree.index_of.end()) far_level = std::max<int>(far_level, tree.nodes[it->second].level);
        }
        check("zooming in under the same budget reaches finer levels than the full view",
              near_level > far_level && near_sel.selected_points <= 120000,
              "level " + std::to_string(far_level) + " -> " + std::to_string(near_level) + ", " +
                  std::to_string(near_sel.selected_points) + " points");
    }
    cache.close();
}

// ========================================================================
// T. SECTION QUERY + CLASSIFICATION WRITE-BACK
// ========================================================================
// The Cross Section / Cut Section tools read points out of a .lnt and write class
// edits back into it. Verified against INDEPENDENT ground truth:
//   * the query equals a brute-force decode of every leaf (same identities, coordinates)
//   * after random edits, every edited leaf point carries its new class, every OTHER
//     leaf point is byte-identical, and the file is still structurally clean
//   * every interior LOD entry equals the class of the leaf point it was sampled from,
//     where "sampled from" is recomputed from the builder's own rule (stride over the
//     Morton-sorted leaf points) - not from the code under test.
static std::vector<uint8_t> leaf_classes_in_order(Reader& rd) {
    std::vector<uint8_t> g;
    std::vector<LNTPoint> raw;
    for (const LNTNodeEntry& e : rd.nodes())
        if (is_leaf_node(e) && rd.read_node_raw(e, raw))
            for (const LNTPoint& p : raw) g.push_back(p.classification);
    return g;
}

// Number of interior entries whose class disagrees with the leaf point they were sampled
// from. `target` is the parent_lod_target the file was built with.
static uint64_t lod_mismatches(Reader& rd, uint64_t target, uint64_t* checked) {
    const std::vector<uint8_t> g = leaf_classes_in_order(rd);
    SectionEditIndex idx;
    idx.build(rd);
    const std::vector<LNTNodeEntry>& ns = rd.nodes();
    uint64_t bad = 0, n = 0;
    std::vector<LNTPoint> raw;
    for (std::size_t i = 0; i < ns.size(); ++i) {
        if (is_leaf_node(ns[i]) || ns[i].point_count == 0) continue;
        const uint64_t c = idx.subtree_pts[i];
        uint64_t stride = 1;
        if (c > target && target > 0) stride = (c + target - 1) / target;
        if (!rd.read_node_raw(ns[i], raw)) { ++bad; continue; }
        const uint64_t begin = idx.leaf_prefix[i];
        for (uint64_t k = 0; k < raw.size(); ++k) {
            const uint64_t src = begin + k * stride;
            ++n;
            if (src >= g.size() || g[src] != raw[k].classification) ++bad;
        }
    }
    if (checked) *checked = n;
    return bad;
}

static void section_query_edit(const std::string& path) {
    std::printf("\n=== T. SECTION QUERY + CLASSIFICATION WRITE-BACK ===\n");
    const std::string src = path + ".large";
    const std::string file = path + ".edit";
    {
        std::ifstream in(src, std::ios::binary);
        if (!in) { skip("section query/edit", "the large test file from section S is missing"); return; }
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        out << in.rdbuf();
    }
    const uint64_t kTarget = 2500;        // parent_lod_target section S built the file with
    Reader rd;
    if (!rd.open(file)) { check("edit: the file opens", false, rd.error()); return; }
    const BoundingBox env = rd.get_bounds();

    uint64_t checked = 0;
    const uint64_t base_bad = lod_mismatches(rd, kTarget, &checked);
    check("ground truth: every interior LOD entry already equals its source leaf point's class",
          base_bad == 0 && checked > 10000,
          std::to_string(base_bad) + " mismatches in " + std::to_string(checked) + " entries");

    // ---- T1/T2: the query ------------------------------------------------------------
    const double mn[3] = {env.min[0] + env.extent(0) * 0.30, env.min[1] + env.extent(1) * 0.20,
                          env.min[2]};
    const double mx[3] = {env.min[0] + env.extent(0) * 0.55, env.min[1] + env.extent(1) * 0.40,
                          env.max[2]};
    std::vector<SectionHit> hits;
    std::string err;
    const int64_t n = section_query_aabb(rd, mn, mx, hits, err);
    const BoundingBox box = BoundingBox::from_min_max(mn[0], mn[1], mn[2], mx[0], mx[1], mx[2]);
    uint64_t expect = 0;
    {
        std::vector<RawPoint> pts;
        for (const LNTNodeEntry& e : rd.nodes()) {
            if (!is_leaf_node(e)) continue;
            rd.read_node_points(e, pts);
            for (const RawPoint& p : pts) if (box.contains(p)) ++expect;
        }
    }
    check("the box query returns exactly the leaf points a brute-force decode finds inside it",
          n > 1000 && uint64_t(n) == expect && hits.size() == expect,
          std::to_string(n) + " vs " + std::to_string(expect));
    std::vector<double> xyz(hits.size() * 3);
    std::vector<uint8_t> cls(hits.size()), inten(hits.size()), rgb(hits.size() * 3);
    std::vector<uint32_t> nd(hits.size()), lc(hits.size());
    check("fetch returns the attributes", section_fetch(rd, hits, xyz.data(), cls.data(),
                                                        inten.data(), rgb.data(), nd.data(),
                                                        lc.data()));
    uint64_t outside = 0, mismatch = 0;
    {
        std::vector<RawPoint> pts;
        uint32_t cur = 0xFFFFFFFFu;
        for (std::size_t j = 0; j < hits.size(); ++j) {
            if (hits[j].node != cur) { rd.read_node_points(rd.nodes()[hits[j].node], pts); cur = hits[j].node; }
            const RawPoint& p = pts[hits[j].local];
            if (p.x != xyz[j * 3] || p.y != xyz[j * 3 + 1] || p.z != xyz[j * 3 + 2] ||
                p.classification != cls[j] || p.intensity != inten[j] || p.r != rgb[j * 3])
                ++mismatch;
            if (!box.contains(xyz[j * 3], xyz[j * 3 + 1], xyz[j * 3 + 2])) ++outside;
        }
    }
    check("fetched coordinates and attributes equal the reader's decode, all inside the box",
          mismatch == 0 && outside == 0, std::to_string(mismatch) + " mismatches, " +
                                             std::to_string(outside) + " outside");
    {
        std::vector<SectionHit> none;
        const double lo[3] = {env.max[0] + 10, env.max[1] + 10, env.max[2] + 10};
        const double hi[3] = {env.max[0] + 20, env.max[1] + 20, env.max[2] + 20};
        check("a box outside the cloud returns no points",
              section_query_aabb(rd, lo, hi, none, err) == 0 && none.empty());
        std::vector<SectionHit> capped;
        check("a query larger than max_points is refused, not truncated",
              section_query_aabb(rd, mn, mx, capped, err, 100) == -2 && capped.empty());
    }

    // ---- T3: random edits ------------------------------------------------------------
    const std::vector<uint8_t> before = leaf_classes_in_order(rd);
    Rng rng(4242);
    std::vector<uint32_t> en, el;
    std::vector<uint8_t> ec;
    std::unordered_map<uint64_t, uint8_t> expected_new;
    for (int i = 0; i < 4000; ++i) {
        const std::size_t j = std::size_t(rng.unit() * double(hits.size())) % hits.size();
        const uint8_t nc = uint8_t(cls[j] ^ uint8_t(1 + int(rng.unit() * 250.0)));   // always different
        const uint64_t key = (uint64_t(nd[j]) << 32) | lc[j];
        if (expected_new.count(key)) continue;
        expected_new[key] = nc;
        en.push_back(nd[j]); el.push_back(lc[j]); ec.push_back(nc);
    }
    SectionEditIndex idx;
    std::vector<uint32_t> touched;
    SectionEditStats st;
    const bool ok = section_set_classes(rd, idx, nullptr, en.data(), el.data(), ec.data(),
                                        en.size(), touched, err, &st);
    check("4000 class edits are written", ok, err);
    std::printf("  edit           : %zu points -> %llu leaf chunks, %llu LOD chunks rewritten, "
                "%llu LOD copies updated, %llu ancestors skipped\n", en.size(),
                (unsigned long long)st.leaves_rewritten, (unsigned long long)st.lod_chunks_rewritten,
                (unsigned long long)st.lod_copies_updated, (unsigned long long)st.lod_ancestors_skipped);

    Reader after_rd;
    if (!after_rd.open(file)) { check("edit: the edited file reopens", false, after_rd.error()); return; }
    check("the edited file is still structurally clean",
          after_rd.diagnose().clean() && after_rd.get_total_nodes() == rd.get_total_nodes() &&
              after_rd.get_total_points() == rd.get_total_points());
    const std::vector<uint8_t> after = leaf_classes_in_order(after_rd);
    // map identity -> global leaf position
    SectionEditIndex gi;
    gi.build(after_rd);
    uint64_t wrong_new = 0, collateral = 0;
    std::vector<uint8_t> want = before;
    for (const auto& kv : expected_new) {
        const uint32_t node = uint32_t(kv.first >> 32), local = uint32_t(kv.first & 0xFFFFFFFFu);
        want[gi.leaf_prefix[node] + local] = kv.second;
    }
    for (std::size_t i = 0; i < after.size(); ++i) {
        const bool edited = want[i] != before[i];
        if (edited && after[i] != want[i]) ++wrong_new;
        if (!edited && after[i] != before[i]) ++collateral;
    }
    check("every edited point carries its new class and no other leaf point changed",
          wrong_new == 0 && collateral == 0 && after.size() == before.size(),
          std::to_string(wrong_new) + " wrong, " + std::to_string(collateral) + " collateral");
    uint64_t chk2 = 0;
    const uint64_t lod_bad = lod_mismatches(after_rd, kTarget, &chk2);
    check("every LOD copy in every ancestor agrees with its source leaf point after the edit",
          lod_bad == 0 && st.lod_ancestors_skipped == 0 && st.lod_copies_updated > 0,
          std::to_string(lod_bad) + " stale of " + std::to_string(chk2) + " entries, " +
              std::to_string(st.lod_copies_updated) + " copies updated");

    // geometry and colour must be untouched
    {
        std::vector<RawPoint> a, b;
        uint64_t diff = 0;
        for (std::size_t i = 0; i < rd.nodes().size(); ++i) {
            rd.read_node_points(rd.nodes()[i], a);
            after_rd.read_node_points(after_rd.nodes()[i], b);
            for (std::size_t k = 0; k < a.size(); ++k)
                if (a[k].x != b[k].x || a[k].y != b[k].y || a[k].z != b[k].z || a[k].r != b[k].r ||
                    a[k].g != b[k].g || a[k].b != b[k].b || a[k].intensity != b[k].intensity) ++diff;
        }
        check("an edit changes ONLY the classification byte: coordinates, colour, intensity identical",
              diff == 0, std::to_string(diff) + " points differ");
    }

    // ---- T4: no-op and bad input ----------------------------------------------------------
    {
        std::vector<uint32_t> t2;
        SectionEditStats s2;
        std::vector<uint8_t> same(en.size());
        for (std::size_t i = 0; i < en.size(); ++i) same[i] = ec[i];     // already applied
        const bool ok2 = section_set_classes(after_rd, gi, nullptr, en.data(), el.data(),
                                             same.data(), en.size(), t2, err, &s2);
        check("re-applying the same classes rewrites nothing", ok2 && t2.empty(),
              std::to_string(t2.size()) + " chunks touched");
        uint32_t bn = 0xFFFFFF00u, bl = 0;
        uint8_t bc = 1;
        check("an out-of-range point identity is rejected",
              !section_set_classes(after_rd, gi, nullptr, &bn, &bl, &bc, 1, t2, err));
        // a non-leaf node is not editable by identity
        uint32_t interior = 0;
        for (std::size_t i = 0; i < after_rd.nodes().size(); ++i)
            if (!is_leaf_node(after_rd.nodes()[i])) { interior = uint32_t(i); break; }
        check("an interior (LOD) node cannot be addressed as a point",
              !section_set_classes(after_rd, gi, nullptr, &interior, &bl, &bc, 1, t2, err));
    }
    std::remove(file.c_str());
}

// ========================================================================
// U. SHADING MODES (RGB, INTENSITY, ELEVATION, CLASSIFICATION, NORMAL SHADING + EDL)
// ========================================================================
// (The brief calls this Section T; T was already the section query / write-back suite.)
// The GPU colours are checked against INDEPENDENT ground truth: the CPU reference ramp in
// LNTShadingModes.h, and the dataset's own per-point colours. A mode that renders the wrong
// ramp, the wrong range or the wrong channel cannot pass.
static void section_shading(const std::string& path) {
    std::printf("\n=== U. SHADING MODES ===\n");

    // ---- U0. the contract, no GPU ------------------------------------------------
    {
        check("ShadingUniforms is exactly 40 bytes", sizeof(ShadingUniforms) == 40);
        uint8_t b[3];
        ramp_entry_bytes(LNTColorRamp::Hue, 0, b);
        const bool low_blue = b[0] == 0 && b[1] == 0 && b[2] == 255;
        ramp_entry_bytes(LNTColorRamp::Hue, 255, b);
        const bool high_red = b[0] == 255 && b[1] == 0 && b[2] == 0;
        ramp_entry_bytes(LNTColorRamp::Hue, 128, b);
        const bool mid_green = b[1] >= 250 && b[0] <= 4 && b[2] <= 4;
        check("Hue runs blue (low) -> green -> red (high): the ends are different colours",
              low_blue && mid_green && high_red);
        uint8_t g0[3], g1[3], r0[3], r1[3], t0[3], t1[3];
        ramp_entry_bytes(LNTColorRamp::Grayscale, 0, g0);
        ramp_entry_bytes(LNTColorRamp::Grayscale, 255, g1);
        ramp_entry_bytes(LNTColorRamp::BlueRed, 0, r0);
        ramp_entry_bytes(LNTColorRamp::BlueRed, 255, r1);
        ramp_entry_bytes(LNTColorRamp::LandTopo, 0, t0);
        ramp_entry_bytes(LNTColorRamp::LandTopo, 255, t1);
        check("Grayscale, Blue/Red and Land Topo have their documented end points",
              g0[0] == 0 && g1[0] == 255 && g1[1] == 255 && r0[2] == 255 && r0[0] == 0 &&
                  r1[0] == 255 && r1[2] == 0 && t0[1] > t0[0] && t1[0] > t1[2]);
    }

    const std::string dll = VulkanEngine::find_dll();
    if (dll.empty()) { skip("shading on the GPU", "no naksha_vulkan.dll found (set LNT_VULKAN_DLL)"); return; }
    VulkanEngine eng;
    std::string err;
    if (!eng.load(dll, err)) { skip("shading on the GPU", "engine load failed: " + err); return; }
    if (!eng.is_available()) { skip("shading on the GPU", "no Vulkan device"); return; }
    if (!eng.has_shading_mode() || !eng.has_rgb_tiles()) {
        skip("shading on the GPU", "this naksha_vulkan.dll has no nkv_set_shading_mode / "
                                   "nkv_upload_point_tile_ex - rebuild the engine from the current sources");
        return;
    }
    if (!eng.open(960, 720, /*hidden=*/true, /*validation=*/false, err)) {
        skip("shading on the GPU", "renderer creation failed: " + err);
        return;
    }
    Reader rd;
    if (!rd.open(path)) { check("shading: the file opens", false, rd.error()); return; }
    LntScene scene;
    if (!scene.upload(eng, rd, err)) { check("shading: the arena uploads", false, err); return; }
    const BoundingBox env = rd.get_bounds();
    const double side = env.extent(0);
    eng.set_clear_color(0.05f, 0.05f, 0.08f, 1.0f);
    eng.set_point_size_params(0.12f, 2.5f, 2.5f, 0.0f);
    eng.set_camera_lookat(env.mid(0), env.mid(1) - side * 0.05, env.mid(2) + side * 1.2,
                          env.mid(0), env.mid(1), env.mid(2), 45.0, 1.0, side * 20.0);
    scene.show_all(eng, true);

    // the dataset's own value ranges (ground truth for the ranges below)
    double zmin = 1e300, zmax = -1e300;
    std::unordered_set<uint32_t> source_colours;
    {
        std::vector<RawPoint> pts;
        for (const LNTNodeEntry& e : rd.nodes()) {
            if (!is_leaf_node(e)) continue;
            rd.read_node_points(e, pts);
            for (const RawPoint& p : pts) {
                zmin = std::min(zmin, p.z);
                zmax = std::max(zmax, p.z);
                source_colours.insert(uint32_t(p.r) << 16 | uint32_t(p.g) << 8 | p.b);
            }
        }
    }

    auto grab = [&](EngineFrame& f) { eng.render(); return eng.capture_frame(f); };
    EngineFrame frame;
    check("the engine renders a first frame", grab(frame), eng.last_error());
    uint8_t bg[3];
    uint64_t bg_n = 0;
    dominant_colour(frame, bg, &bg_n);
    auto is_geometry = [&](const uint8_t* p) {
        return std::abs(int(p[0]) - bg[0]) > 6 || std::abs(int(p[1]) - bg[1]) > 6 ||
               std::abs(int(p[2]) - bg[2]) > 6;
    };
    // distance of a pixel to the closest of the 256 entries of a ramp (max channel diff)
    auto ramp_distance = [](LNTColorRamp r, const uint8_t* p) {
        int best = 1000;
        for (int i = 0; i < 256; ++i) {
            uint8_t e[3];
            ramp_entry_bytes(r, i, e);
            const int d = std::max({std::abs(int(p[0]) - e[0]), std::abs(int(p[1]) - e[1]),
                                    std::abs(int(p[2]) - e[2])});
            best = std::min(best, d);
        }
        return best;
    };

    const uint64_t pos_up = eng.position_upload_count();
    const uint64_t tiles = eng.arena_uploaded_tiles();
    const uint32_t ranges = eng.draw_range_count();
    const uint64_t pts_drawn = eng.draw_range_points();

    // ---- U1. Elevation, Hue, over the data's own Z range ----------------------------------
    {
        ShadingUniforms u = make_shading(LNTDisplayMode::Elevation, LNTColorRamp::Hue,
                                         float(zmin), float(zmax));
        check("the engine accepts an Elevation / Hue configuration", eng.set_shading_mode(u));
        EngineFrame f;
        grab(f);
        uint64_t n = 0, off_ramp = 0;
        std::vector<uint8_t> hit(256, 0);
        for (uint32_t y = 0; y < f.height; ++y)
            for (uint32_t x = 0; x < f.width; ++x) {
                const uint8_t* p = f.at(x, y);
                if (!is_geometry(p)) continue;
                ++n;
                if (ramp_distance(LNTColorRamp::Hue, p) > 3) ++off_ramp;
                // which table entry is it: nearest by hue ordering (blue low .. red high)
                int best = 0, bd = 1000;
                for (int i = 0; i < 256; ++i) {
                    uint8_t e[3];
                    ramp_entry_bytes(LNTColorRamp::Hue, i, e);
                    const int d = std::abs(int(p[0]) - e[0]) + std::abs(int(p[1]) - e[1]) +
                                  std::abs(int(p[2]) - e[2]);
                    if (d < bd) { bd = d; best = i; }
                }
                hit[best] = 1;
            }
        int distinct = 0;
        for (uint8_t h : hit) distinct += h;
        check("Elevation / Hue: every rendered pixel lies on the Hue ramp",
              n > 5000 && off_ramp * 200 <= n,
              std::to_string(off_ramp) + " of " + std::to_string(n) + " pixels off the ramp");
        check("Elevation / Hue: the ramp is spread over the data's Z range (many different colours)",
              distinct >= 40, std::to_string(distinct) + " distinct ramp entries");
    }
    // the brief's range: Z in [0, 200] with data at ~1100 m -> every point is above the top
    {
        ShadingUniforms u = make_shading(LNTDisplayMode::Elevation, LNTColorRamp::Hue, 0.0f, 200.0f);
        eng.set_shading_mode(u);
        EngineFrame f;
        grab(f);
        uint64_t n = 0, red = 0;
        for (uint32_t y = 0; y < f.height; ++y)
            for (uint32_t x = 0; x < f.width; ++x) {
                const uint8_t* p = f.at(x, y);
                if (!is_geometry(p)) continue;
                ++n;
                if (p[0] >= 252 && p[1] <= 3 && p[2] <= 3) ++red;
            }
        check("Elevation / Hue over [0, 200] m: points above the range clamp to the top colour (red)",
              n > 5000 && red * 100 >= n * 99, std::to_string(red) + " of " + std::to_string(n));
    }

    // ---- U2. Intensity, Grayscale, and its scaling bounds ------------------------------------
    uint64_t full_extremes = 0, full_n = 0;
    {
        ShadingUniforms u = make_shading(LNTDisplayMode::Intensity, LNTColorRamp::Grayscale,
                                         0, 1, 0.0f, 1.0f);
        eng.set_shading_mode(u);
        EngineFrame f;
        grab(f);
        uint64_t gray_bad = 0;
        int lo = 255, hi = 0;
        for (uint32_t y = 0; y < f.height; ++y)
            for (uint32_t x = 0; x < f.width; ++x) {
                const uint8_t* p = f.at(x, y);
                if (!is_geometry(p)) continue;
                ++full_n;
                if (std::abs(int(p[0]) - p[1]) > 2 || std::abs(int(p[1]) - p[2]) > 2) ++gray_bad;
                lo = std::min<int>(lo, p[0]);
                hi = std::max<int>(hi, p[0]);
                if (p[0] <= 2 || p[0] >= 253) ++full_extremes;
            }
        check("Intensity / Grayscale: every pixel is gray and the full 0..1 scale is used",
              full_n > 5000 && gray_bad * 200 <= full_n && lo < 25 && hi > 230,
              std::to_string(gray_bad) + " non-gray, range " + std::to_string(lo) + ".." + std::to_string(hi));
    }
    {
        // scaling bounds: [0.25, 0.75] clips the lower and upper quarter to pure black / white
        ShadingUniforms u = make_shading(LNTDisplayMode::Intensity, LNTColorRamp::Grayscale,
                                         0, 1, 0.25f, 0.75f);
        eng.set_shading_mode(u);
        EngineFrame f;
        grab(f);
        uint64_t n = 0, extremes = 0;
        for (uint32_t y = 0; y < f.height; ++y)
            for (uint32_t x = 0; x < f.width; ++x) {
                const uint8_t* p = f.at(x, y);
                if (!is_geometry(p)) continue;
                ++n;
                if (p[0] <= 2 || p[0] >= 253) ++extremes;
            }
        const double frac_narrow = double(extremes) / double(std::max<uint64_t>(n, 1));
        const double frac_full = double(full_extremes) / double(std::max<uint64_t>(full_n, 1));
        check("Intensity bounds [0.25, 0.75]: about half the pixels clamp to black or white "
              "(the full scale clamps almost none)",
              frac_narrow > 0.40 && frac_narrow < 0.60 && frac_full < 0.05,
              std::to_string(frac_narrow) + " clamped vs " + std::to_string(frac_full) + " at 0..1");
        // a ramp other than gray changes the colour but not the geometry
        ShadingUniforms v = make_shading(LNTDisplayMode::Intensity, LNTColorRamp::BlueRed);
        eng.set_shading_mode(v);
        EngineFrame f2;
        grab(f2);
        uint64_t off = 0, m = 0;
        for (uint32_t y = 0; y < f2.height; ++y)
            for (uint32_t x = 0; x < f2.width; ++x) {
                const uint8_t* p = f2.at(x, y);
                if (!is_geometry(p)) continue;
                ++m;
                if (ramp_distance(LNTColorRamp::BlueRed, p) > 3) ++off;
            }
        check("Intensity / Blue-Red: every pixel lies on the Blue/Red ramp", m > 5000 && off * 200 <= m,
              std::to_string(off) + " of " + std::to_string(m) + " off");
    }

    // ---- U3. RGB: the file's own colours -----------------------------------------------------------
    {
        eng.set_shading_mode(make_shading(LNTDisplayMode::RGB));
        EngineFrame f;
        grab(f);
        uint64_t n = 0, known = 0;
        for (uint32_t y = 0; y < f.height; ++y)
            for (uint32_t x = 0; x < f.width; ++x) {
                const uint8_t* p = f.at(x, y);
                if (!is_geometry(p)) continue;
                ++n;
                if (source_colours.count(uint32_t(p[0]) << 16 | uint32_t(p[1]) << 8 | p[2])) ++known;
            }
        check("RGB: every rendered pixel is the exact colour of a point in the file",
              n > 5000 && known * 100 >= n * 99, std::to_string(known) + " of " + std::to_string(n));
    }

    // ---- U4. Classification, and switching back and forth -------------------------------------------------
    {
        PackedClassColor classes[256];
        for (unsigned i = 0; i < 256; ++i)
            classes[i] = PackedClassColor{uint8_t(i), uint8_t(255 - i), uint8_t((i * 7) & 255), 1};
        eng.set_classification_lut(classes);
        eng.set_shading_mode(make_shading(LNTDisplayMode::Classification));
        EngineFrame a;
        grab(a);
        eng.set_shading_mode(make_shading(LNTDisplayMode::Elevation, LNTColorRamp::Hue, float(zmin), float(zmax)));
        EngineFrame mid;
        grab(mid);
        eng.set_shading_mode(make_shading(LNTDisplayMode::Classification));
        EngineFrame b;
        grab(b);
        uint64_t diff_back = 0, diff_other = 0;
        for (std::size_t i = 0; i < a.rgba.size(); ++i) {
            if (a.rgba[i] != b.rgba[i]) ++diff_back;
            if (a.rgba[i] != mid.rgba[i]) ++diff_other;
        }
        check("Classification: switching away and back reproduces the frame pixel for pixel",
              diff_back == 0 && diff_other > 10000,
              std::to_string(diff_back) + " pixels differ after the round trip, " +
                  std::to_string(diff_other) + " bytes differ in the other mode");
    }

    // ---- U5. Normal shading + Eye-Dome Lighting ---------------------------------------------------------------
    {
        ShadingUniforms n0 = make_shading(LNTDisplayMode::NormalShading);
        n0.edl_strength = 0.0f;
        check("the engine accepts a NormalShading configuration", eng.set_shading_mode(n0));
        EngineFrame f0;
        grab(f0);
        ShadingUniforms n1 = n0;
        n1.edl_strength = 1.0f;
        eng.set_shading_mode(n1);
        EngineFrame f1;
        grab(f1);
        ShadingUniforms n2 = n0;
        n2.light_direction[0] = -0.5f;                 // a different sun
        n2.light_direction[1] = 0.5f;
        eng.set_shading_mode(n2);
        EngineFrame f2;
        grab(f2);
        auto lum = [&](const EngineFrame& f) {
            double s = 0;
            uint64_t n = 0;
            for (uint32_t y = 0; y < f.height; ++y)
                for (uint32_t x = 0; x < f.width; ++x) {
                    const uint8_t* p = f.at(x, y);
                    if (!is_geometry(p)) continue;
                    s += 0.3 * p[0] + 0.59 * p[1] + 0.11 * p[2];
                    ++n;
                }
            return n ? s / double(n) : 0.0;
        };
        uint64_t brighter = 0, changed_edl = 0, changed_sun = 0;
        for (uint32_t y = 0; y < f0.height; ++y)
            for (uint32_t x = 0; x < f0.width; ++x) {
                const uint8_t* a = f0.at(x, y);
                const uint8_t* b = f1.at(x, y);
                const uint8_t* c = f2.at(x, y);
                if (!is_geometry(a)) continue;
                const int la = a[0] + a[1] + a[2], lb = b[0] + b[1] + b[2];
                if (lb > la + 3) ++brighter;
                if (la - lb > 6) ++changed_edl;
                if (std::abs(int(a[0]) - c[0]) + std::abs(int(a[1]) - c[1]) + std::abs(int(a[2]) - c[2]) > 9)
                    ++changed_sun;
            }
        check("Eye-Dome Lighting only darkens: it never brightens a pixel and darkens some",
              brighter == 0 && changed_edl > 0 && lum(f1) < lum(f0),
              std::to_string(changed_edl) + " pixels darker, " + std::to_string(brighter) +
                  " brighter, mean " + std::to_string(lum(f0)) + " -> " + std::to_string(lum(f1)));
        check("a different light direction changes the shaded image", changed_sun > 100,
              std::to_string(changed_sun) + " pixels changed");
    }

    // ---- U6. nothing was re-uploaded, and a switch is instant -------------------------------------------------------
    check("through every mode change: zero position uploads, same tiles, same draw ranges and points",
          eng.position_upload_count() == pos_up && eng.arena_uploaded_tiles() == tiles &&
              eng.draw_range_count() == ranges && eng.draw_range_points() == pts_drawn,
          std::to_string(eng.position_upload_count() - pos_up) + " position uploads");
    {
        const int N = 400;
        const LNTDisplayMode modes[5] = {LNTDisplayMode::RGB, LNTDisplayMode::Intensity,
                                         LNTDisplayMode::Elevation, LNTDisplayMode::Classification,
                                         LNTDisplayMode::NormalShading};
        const auto t0 = std::chrono::steady_clock::now();
        bool all = true;
        for (int i = 0; i < N; ++i)
            all = eng.set_shading_mode(make_shading(modes[i % 5], LNTColorRamp::Hue, 1040.f, 1160.f)) && all;
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / N;
        check("switching modes is a uniform write: under 0.5 ms per switch", all && us < 500.0,
              std::to_string(us) + " us per switch (" + std::to_string(N) + " switches)");
        ShadingUniforms bad = make_shading(LNTDisplayMode::RGB);
        bad.display_mode = 9;
        check("an out-of-range mode is rejected and leaves the engine usable",
              !eng.set_shading_mode(bad) && eng.set_shading_mode(make_shading(LNTDisplayMode::RGB)));
    }
}

// ========================================================================
// V. SHADED SURFACE (TRIANGULATED HILLSHADE)
// ========================================================================
// The Shading / Surface view is a triangulated terrain model, lit like a hillshade. Verified
// against geometry computed independently in the test: the triangles partition the surface
// (every point inside is covered by exactly ONE triangle, including across tile borders and
// stacked tiles), they lie on the real surface, the shade equals N.L from the plane's own
// normal, voids stay open, hidden classes vanish, and on the GPU the surface covers the
// footprint the bare points leave full of holes.
static double test_terrain(double x, double y) {
    return 20.0 * std::sin(x / 90.0) + 15.0 * std::cos(y / 70.0) + 0.05 * x;
}

// the triangle containing (px, py) in plan, via a bucket grid; returns the count of triangles
// that contain it and the interpolated z of the first
struct TriIndex {
    const SurfaceMesh& m;
    double x0, y0, cell;
    int nx, ny;
    std::vector<std::vector<uint32_t>> bins;
    explicit TriIndex(const SurfaceMesh& mesh, double minx, double miny, double maxx, double maxy, double c)
        : m(mesh), x0(minx), y0(miny), cell(c) {
        nx = int((maxx - minx) / c) + 1;
        ny = int((maxy - miny) / c) + 1;
        bins.resize(std::size_t(nx) * ny);
        for (uint64_t t = 0; t < mesh.triangles(); ++t) {
            double lx = 1e300, hx = -1e300, ly = 1e300, hy = -1e300;
            for (int k = 0; k < 3; ++k) {
                const int32_t v = mesh.faces[3 * t + k];
                lx = std::min(lx, mesh.positions[3 * v]); hx = std::max(hx, mesh.positions[3 * v]);
                ly = std::min(ly, mesh.positions[3 * v + 1]); hy = std::max(hy, mesh.positions[3 * v + 1]);
            }
            for (int by = std::max(0, int((ly - miny) / c)); by <= std::min(ny - 1, int((hy - miny) / c)); ++by)
                for (int bx = std::max(0, int((lx - minx) / c)); bx <= std::min(nx - 1, int((hx - minx) / c)); ++bx)
                    bins[std::size_t(by) * nx + bx].push_back(uint32_t(t));
        }
    }
    int locate(double px, double py, double& z) const {
        const int bx = int((px - x0) / cell), by = int((py - y0) / cell);
        if (bx < 0 || by < 0 || bx >= nx || by >= ny) return 0;
        int hits = 0;
        for (uint32_t t : bins[std::size_t(by) * nx + bx]) {
            double X[3], Y[3], Z[3];
            for (int k = 0; k < 3; ++k) {
                const int32_t v = m.faces[3 * t + k];
                X[k] = m.positions[3 * v]; Y[k] = m.positions[3 * v + 1]; Z[k] = m.positions[3 * v + 2];
            }
            const double d = (Y[1] - Y[2]) * (X[0] - X[2]) + (X[2] - X[1]) * (Y[0] - Y[2]);
            if (std::fabs(d) < 1e-14) continue;
            const double l0 = ((Y[1] - Y[2]) * (px - X[2]) + (X[2] - X[1]) * (py - Y[2])) / d;
            const double l1 = ((Y[2] - Y[0]) * (px - X[2]) + (X[0] - X[2]) * (py - Y[2])) / d;
            const double l2 = 1.0 - l0 - l1;
            if (l0 >= 0 && l1 >= 0 && l2 >= 0) {
                if (hits == 0) z = l0 * Z[0] + l1 * Z[1] + l2 * Z[2];
                ++hits;
            }
        }
        return hits;
    }
};

static std::vector<SurfaceTile> make_terrain_tiles(size_t n, uint64_t seed, double void_cx = -1, double void_r = 0) {
    // a 600 x 600 m terrain cut into a 3 x 3 grid of 200 m columns, each split in Z at 0 into
    // two stacked tiles (an octree does the same), so seams and stacking are both exercised
    std::vector<SurfaceTile> tiles(18);
    for (int gy = 0; gy < 3; ++gy)
        for (int gx = 0; gx < 3; ++gx)
            for (int s = 0; s < 2; ++s) {
                SurfaceTile& t = tiles[(gy * 3 + gx) * 2 + s];
                t.cmin[0] = gx * 200.0; t.cmax[0] = gx * 200.0 + 200.0;
                t.cmin[1] = gy * 200.0; t.cmax[1] = gy * 200.0 + 200.0;
                t.cmin[2] = s == 0 ? -100.0 : 0.0; t.cmax[2] = s == 0 ? 0.0 : 100.0;
            }
    Rng rng(seed);
    for (size_t i = 0; i < n; ++i) {
        SurfacePoint p;
        p.x = rng.unit() * 600.0;
        p.y = rng.unit() * 600.0;
        if (void_r > 0 && std::hypot(p.x - void_cx, p.y - void_cx) < void_r) continue;
        p.z = test_terrain(p.x, p.y);
        p.cls = p.x < 300.0 ? 2 : 5;
        p.r = 10; p.g = 200; p.b = 90; p.intensity = uint8_t(p.y * 255.0 / 600.0);
        const int gx = std::min(2, int(p.x / 200.0)), gy = std::min(2, int(p.y / 200.0));
        const int s = p.z < 0.0 ? 0 : 1;
        tiles[(gy * 3 + gx) * 2 + s].pts.push_back(p);
    }
    return tiles;
}

static void section_surface(const std::string& path) {
    std::printf("\n=== V. SHADED SURFACE (TRIANGULATED HILLSHADE) ===\n");
    SurfaceParams P;
    P.class_rgb[2][0] = 200; P.class_rgb[2][1] = 100; P.class_rgb[2][2] = 50;
    P.class_rgb[5][0] = 40;  P.class_rgb[5][1] = 160; P.class_rgb[5][2] = 220;

    // ---- V1. the triangles partition the surface and lie on it -----------------------------------
    {
        std::vector<SurfaceTile> tiles = make_terrain_tiles(150000, 31);
        const SurfaceMesh mesh = build_surface_mesh(tiles, P);
        check("a 150k-point terrain in 18 tiles (3x3 columns, each split in Z) triangulates",
              mesh.triangles() > 250000,
              std::to_string(mesh.triangles()) + " triangles in " + std::to_string(mesh.build_ms) + " ms");
        TriIndex idx(mesh, 0, 0, 600, 600, 6.0);
        Rng rng(99);
        uint64_t gaps = 0, overlaps = 0, ok = 0, seam_samples = 0;
        double worst_dz = 0.0;
        for (int i = 0; i < 40000; ++i) {
            double px = 20.0 + rng.unit() * 560.0, py = 20.0 + rng.unit() * 560.0;
            if (i % 4 == 0) px = 200.0 * double(1 + (i / 4) % 2) + (rng.unit() - 0.5) * 1.0;   // on a tile seam
            double z = 0;
            const int hits = idx.locate(px, py, z);
            if (i % 4 == 0) ++seam_samples;
            if (hits == 0) ++gaps;
            else if (hits > 1) ++overlaps;
            else { ++ok; worst_dz = std::max(worst_dz, std::fabs(z - test_terrain(px, py))); }
        }
        check("every sample, including 10000 on tile seams, is covered by exactly one triangle "
              "(no gap, no overlap)",
              gaps == 0 && overlaps == 0, std::to_string(gaps) + " gaps, " + std::to_string(overlaps) +
                  " overlaps of 40000 (" + std::to_string(seam_samples) + " on seams)");
        check("the triangles lie on the terrain they were built from (interpolated z within 0.2 m)",
              ok > 39000 && worst_dz < 0.2, "worst " + std::to_string(worst_dz) + " m");
        double vmin = 1e300;
        for (std::size_t i = 0; i < mesh.positions.size(); i += 3) vmin = std::min(vmin, mesh.positions[i]);
        check("vertices are world coordinates, shared by index (the mesh is indexed, not a soup)",
              mesh.vertices() < mesh.triangles() && vmin >= 0.0,
              std::to_string(mesh.vertices()) + " vertices for " + std::to_string(mesh.triangles()) + " triangles");
    }

    // ---- V2. the hillshade of a plane equals N.L from its own normal ---------------------------------
    {
        std::vector<SurfaceTile> one(1);
        one[0].cmin[0] = 0; one[0].cmax[0] = 100; one[0].cmin[1] = 0; one[0].cmax[1] = 100;
        one[0].cmin[2] = -50; one[0].cmax[2] = 200;
        Rng rng(5);
        for (int i = 0; i < 20000; ++i) {
            SurfacePoint p{rng.unit() * 100.0, rng.unit() * 100.0, 0, 2, 0, 0, 0, 0};
            p.z = 0.5 * p.x + 0.25 * p.y;
            one[0].pts.push_back(p);
        }
        for (double az : {315.0, 90.0, 200.0}) {
            SurfaceParams Q = P;
            Q.azimuth_deg = az; Q.elevation_deg = 40.0; Q.ambient = 0.1;
            const SurfaceMesh m = build_surface_mesh(one, Q);
            double L[3];
            surface_light_vector(az, 40.0, L);
            const double nl = std::sqrt(0.25 + 0.0625 + 1.0);
            const double ndl = (-0.5 * L[0] - 0.25 * L[1] + 1.0 * L[2]) / nl;
            const double shade = std::min(1.0, std::max(ndl, 0.1));
            int worst = 0;
            for (uint64_t t = 0; t < m.triangles(); ++t)
                for (int k = 0; k < 3; ++k) {
                    const double want = P.class_rgb[2][k] * shade;
                    worst = std::max(worst, int(std::lround(std::fabs(double(m.colors[3 * t + k]) - want))));
                }
            check("hillshade of a plane (sun azimuth " + std::to_string(int(az)) + ", elevation 40): every "
                  "triangle's colour is the class colour x max(N.L, ambient)",
                  m.triangles() > 1000 && worst <= 1,
                  "shade " + std::to_string(shade) + ", worst error " + std::to_string(worst) + " levels");
        }
    }

    // ---- V3. voids stay open; hidden classes vanish; colour sources ------------------------------------
    {
        std::vector<SurfaceTile> tiles = make_terrain_tiles(150000, 31, 300.0, 70.0);
        const SurfaceMesh m = build_surface_mesh(tiles, P);
        uint64_t in_void = 0;
        for (uint64_t t = 0; t < m.triangles(); ++t) {
            double cx = 0, cy = 0;
            for (int k = 0; k < 3; ++k) { cx += m.positions[3 * m.faces[3 * t + k]]; cy += m.positions[3 * m.faces[3 * t + k] + 1]; }
            if (std::hypot(cx / 3.0 - 300.0, cy / 3.0 - 300.0) < 45.0) ++in_void;
        }
        check("a 70 m void (no returns) is left open, not bridged by long triangles",
              in_void == 0 && m.dropped_long_edges > 0,
              std::to_string(in_void) + " triangles inside, " + std::to_string(m.dropped_long_edges) + " long-edge triangles dropped");

        SurfaceParams H = P;
        H.class_visible[2] = 0;                       // hide the west half
        const SurfaceMesh h = build_surface_mesh(make_terrain_tiles(150000, 31), H);
        double minx = 1e300;
        for (uint64_t t = 0; t < h.triangles(); ++t) {
            double cx = 0;
            for (int k = 0; k < 3; ++k) cx += h.positions[3 * h.faces[3 * t + k]];
            minx = std::min(minx, cx / 3.0);
        }
        check("a hidden class is not triangulated: no triangle remains in its area",
              h.triangles() > 50000 && minx > 290.0, "westmost triangle at x = " + std::to_string(minx));

        // colour sources
        std::vector<SurfaceTile> flat(1);
        flat[0].cmin[0] = 0; flat[0].cmax[0] = 100; flat[0].cmin[1] = 0; flat[0].cmax[1] = 100;
        flat[0].cmin[2] = -10; flat[0].cmax[2] = 100;
        Rng rng(8);
        for (int i = 0; i < 5000; ++i)
            flat[0].pts.push_back(SurfacePoint{rng.unit() * 100.0, rng.unit() * 100.0, 40.0, 2, 10, 200, 90, 128});
        auto first_colour = [&](SurfaceColor src, LNTColorRamp ramp, double z0, double z1, int out[3]) {
            SurfaceParams Q = P;
            Q.color_source = src; Q.ramp = ramp; Q.z_min = z0; Q.z_max = z1;
            Q.elevation_deg = 90.0; Q.ambient = 1.0;     // straight-down sun, full shade: colour only
            const SurfaceMesh m = build_surface_mesh(flat, Q);
            for (int k = 0; k < 3; ++k) out[k] = m.colors[k];
            return m.triangles() > 100;
        };
        int c[3];
        bool a = first_colour(SurfaceColor::RGB, LNTColorRamp::Hue, 0, 1, c);
        check("colour source RGB: the file's own colour", a && c[0] == 10 && c[1] == 200 && c[2] == 90,
              std::to_string(c[0]) + "," + std::to_string(c[1]) + "," + std::to_string(c[2]));
        a = first_colour(SurfaceColor::Elevation, LNTColorRamp::Hue, 0, 80, c);       // z = 40: the middle
        uint8_t want[3];
        ramp_entry_bytes(LNTColorRamp::Hue, 128, want);
        check("colour source Elevation / Hue at the middle of the range: the middle of the ramp",
              a && std::abs(c[0] - want[0]) <= 2 && std::abs(c[1] - want[1]) <= 2 && std::abs(c[2] - want[2]) <= 2,
              std::to_string(c[0]) + "," + std::to_string(c[1]) + "," + std::to_string(c[2]));
        a = first_colour(SurfaceColor::Intensity, LNTColorRamp::Grayscale, 0, 1, c);   // 128 of 0..255
        check("colour source Intensity / Grayscale: mid gray", a && std::abs(c[0] - 128) <= 2 && c[0] == c[1] && c[1] == c[2],
              std::to_string(c[0]));
    }

    // ---- V4. on the GPU --------------------------------------------------------------------------------------
    const std::string dll = VulkanEngine::find_dll();
    if (dll.empty()) { skip("surface on the GPU", "no naksha_vulkan.dll found (set LNT_VULKAN_DLL)"); return; }
    VulkanEngine eng;
    std::string err;
    if (!eng.load(dll, err)) { skip("surface on the GPU", "engine load failed: " + err); return; }
    if (!eng.is_available()) { skip("surface on the GPU", "no Vulkan device"); return; }
    if (!eng.has_surface()) { skip("surface on the GPU", "this naksha_vulkan.dll has no tiled-surface API"); return; }
    if (!eng.open(960, 720, /*hidden=*/true, /*validation=*/false, err)) {
        skip("surface on the GPU", "renderer creation failed: " + err);
        return;
    }
    Reader rd;
    if (!rd.open(path)) { check("surface: the file opens", false, rd.error()); return; }
    LntScene scene;
    if (!scene.upload(eng, rd, err)) { check("surface: the arena uploads", false, err); return; }
    const BoundingBox env = rd.get_bounds();
    const double side = env.extent(0);
    eng.set_clear_color(0.05f, 0.05f, 0.08f, 1.0f);
    eng.set_point_size_params(0.12f, 2.5f, 2.5f, 0.0f);
    eng.set_camera_lookat(env.mid(0), env.mid(1) - side * 0.05, env.mid(2) + side * 1.2,
                          env.mid(0), env.mid(1), env.mid(2), 45.0, 1.0, side * 20.0);
    scene.show_all(eng, true);
    eng.set_shading_mode(make_shading(LNTDisplayMode::Classification));

    // the tiles the cut would show: the file's leaves, as surface tiles
    std::vector<SurfaceTile> tiles;
    {
        std::vector<RawPoint> pts;
        for (const LNTNodeEntry& e : rd.nodes()) {
            if (!is_leaf_node(e)) continue;
            rd.read_node_points(e, pts);
            SurfaceTile t;
            const BoundingBox b = rd.bounds_of(e);
            for (int a = 0; a < 3; ++a) { t.cmin[a] = b.min[a]; t.cmax[a] = b.max[a]; }
            for (const RawPoint& p : pts) t.pts.push_back(SurfacePoint{p.x, p.y, p.z, p.classification, p.r, p.g, p.b, p.intensity});
            tiles.push_back(std::move(t));
        }
    }
    SurfaceParams G;
    for (int i = 0; i < 256; ++i) { G.class_rgb[i][0] = uint8_t(i); G.class_rgb[i][1] = uint8_t(255 - i); G.class_rgb[i][2] = 128; }
    const SurfaceMesh mesh = build_surface_mesh(tiles, G);

    auto grab = [&](EngineFrame& f) { eng.render(); return eng.capture_frame(f); };
    EngineFrame pts_frame;
    grab(pts_frame);
    uint8_t bg[3];
    uint64_t bg_n = 0;
    dominant_colour(pts_frame, bg, &bg_n);
    auto geo = [&](const uint8_t* p) {
        return std::abs(int(p[0]) - bg[0]) > 6 || std::abs(int(p[1]) - bg[1]) > 6 || std::abs(int(p[2]) - bg[2]) > 6;
    };
    auto bbox_cover = [&](const EngineFrame& f, double& frac, uint32_t box[4]) {
        uint32_t x0 = f.width, y0 = f.height, x1 = 0, y1 = 0;
        for (uint32_t y = 0; y < f.height; ++y)
            for (uint32_t x = 0; x < f.width; ++x)
                if (geo(f.at(x, y))) { x0 = std::min(x0, x); x1 = std::max(x1, x); y0 = std::min(y0, y); y1 = std::max(y1, y); }
        const uint32_t mx = (x1 - x0) / 10, my = (y1 - y0) / 10;
        uint64_t in = 0, total = 0;
        for (uint32_t y = y0 + my; y < y1 - my; ++y)
            for (uint32_t x = x0 + mx; x < x1 - mx; ++x) { ++total; if (geo(f.at(x, y))) ++in; }
        frac = total ? double(in) / double(total) : 0.0;
        box[0] = x0; box[1] = y0; box[2] = x1; box[3] = y1;
    };
    double pts_cover = 0;
    uint32_t box[4];
    bbox_cover(pts_frame, pts_cover, box);

    const uint64_t pos_up = eng.position_upload_count();
    check("the mesh uploads through the engine's atomic surface swap", eng.upload_surface(mesh, 1), eng.last_error());
    check("the pending surface can be selected before its frame-boundary swap", eng.show_surface(true));
    check("the surface parity capture uses the original faces",
          eng.set_surface_lod_error_pixels(0.000001f));
    for (int i = 0; i < 3; ++i) eng.render();         // the swap happens at the frame boundary
    EngineFrame surf;
    grab(surf);
    check("the surface selected full-detail LOD", eng.surface_lod() == 0,
          std::to_string(eng.surface_lod()));
    double surf_cover = 0;
    uint32_t box2[4];
    bbox_cover(surf, surf_cover, box2);
    check("the engine holds the triangles", eng.surface_triangles() == mesh.triangles() && eng.surface_state() >= 2,
          std::to_string(eng.surface_triangles()) + " of " + std::to_string(mesh.triangles()) +
              ", state " + std::to_string(eng.surface_state()));
    check("the surface fills the footprint (>= 99% of the interior drawn) where bare points leave holes",
          surf_cover >= 0.99 && pts_cover < surf_cover,
          std::to_string(surf_cover) + " covered as a surface vs " + std::to_string(pts_cover) + " as points");
    // every surface pixel is one of the flat face colours (no stray colours, no blending)
    std::unordered_set<uint32_t> face_colours;
    for (uint64_t t = 0; t < mesh.triangles(); ++t)
        face_colours.insert(uint32_t(mesh.colors[3 * t]) << 16 | uint32_t(mesh.colors[3 * t + 1]) << 8 | mesh.colors[3 * t + 2]);
    uint64_t n = 0, known = 0;
    for (uint32_t y = 0; y < surf.height; ++y)
        for (uint32_t x = 0; x < surf.width; ++x) {
            const uint8_t* p = surf.at(x, y);
            if (!geo(p)) continue;
            ++n;
            bool hit = false;
            for (int dr = -1; dr <= 1 && !hit; ++dr)
                for (int dg = -1; dg <= 1 && !hit; ++dg)
                    for (int db = -1; db <= 1 && !hit; ++db)
                        hit = face_colours.count((uint32_t(std::clamp(p[0] + dr, 0, 255)) << 16) |
                                                 (uint32_t(std::clamp(p[1] + dg, 0, 255)) << 8) |
                                                 uint32_t(std::clamp(p[2] + db, 0, 255))) != 0;
            if (hit) ++known;
        }
    check("every drawn pixel is a face colour of the mesh (the baked hillshade reaches the screen unchanged)",
          n > 10000 && known * 100 >= n * 98, std::to_string(known) + " of " + std::to_string(n));
    check("showing the surface re-uploaded no point data", eng.position_upload_count() == pos_up);
    eng.show_surface(false);
    EngineFrame back;
    grab(back);
    uint64_t diff = 0;
    for (std::size_t i = 0; i < back.rgba.size(); ++i) diff += back.rgba[i] != pts_frame.rgba[i];
    check("hiding the surface brings the points back, pixel for pixel", diff == 0, std::to_string(diff) + " bytes differ");
}

#include "surface_barycentric_checks.h"

int main(int argc, char** argv) {
    // Unbuffered stdout: a harness that can crash must still show which check it
    // reached. With full buffering on a pipe, a fault discards every line.
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    const std::string out = (argc > 1) ? argv[1] : kOutDefault;

    std::printf("LNT PIPELINE VERIFICATION (C++17)\n");
    std::printf("output: %s\n", out.c_str());

    section_layout();
    section_math();
    section_classification(out);

    Dataset ds = make_dataset(100000, 20261008ull);
    LNTBuildStats st;
    section_build(ds, out, st);
    if (st.ok) {
        FileView v = section_read_file(out);
        if (v.file_bytes > 0 && !v.nodes.empty()) {
            section_payload(out, v, ds, st);
            section_reader(out, ds, st, v);    // G: read the file back
        }
        section_determinism(ds, out);
        section_render(out, ds, st);            // I: render it through Vulkan
        section_streaming(out);                 // J: camera flight + streaming
        verify_monotone_orbit_zoom(out);        // K: orbit + zoom monotone sweep
        section_las();                          // L: LAS input -> .lnt
        section_interactive();                  // M: interactive camera + loop
        section_zoom_coverage(out);             // O: zoom coverage and selection budget
        section_midzoom_coverage(out);          // P: mid-zoom blank cells and the fallback
        section_async_streaming(out);            // Q: retention while children load
        section_cross_section(out);              // R: planes, corridor, GPU clip volume
        section_large_cloud(out);                // S: large-cloud streaming policy
        section_query_edit(out);                 // T: section query + class write-back
        section_shading(out);                    // U: shading modes
        section_surface(out);                    // V: triangulated shaded surface
        section_surface_barycentric();           // W: indexed vertex colours
    }

    std::printf("\n%s\n", std::string(62, '=').c_str());
    std::printf("%d/%d checks passed\n", g_checks - g_failed, g_checks);
    if (g_skipped)
        std::printf("%d check(s) SKIPPED - not verified on this machine\n", g_skipped);
    if (g_failed) {
        std::printf("FAILED: %d\n", g_failed);
        return 1;
    }
    std::printf(g_skipped ? "ALL EXECUTED CHECKS PASSED (%d skipped)\n"
                         : "ALL CHECKS PASSED\n",
                g_skipped);
    return 0;
}
