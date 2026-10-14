// engine_probe.cpp - can this machine drive naksha_vulkan.dll from the native
// .lnt pipeline at all?
//
// This is the feasibility gate for the render path: it proves, step by step and
// printing as it goes, that a MinGW-compiled binary can (1) resolve the engine's
// C ABI out of the MSVC-built DLL, (2) create a device on a hidden window,
// (3) reserve the stable-offset arena and upload node tiles, (4) select a subset
// with draw ranges, and (5) READ BACK REAL PIXELS with the offscreen capture.
//
// Step 5 is the one that matters. A draw-range change that does not change the
// image is not culling, it is a counter that moved - so this probe captures the
// frame twice, once with a range set and once with none, and compares pixels.
//
// Build:
//   g++ -O3 -std=c++17 engine_probe.cpp -o engine_probe
// Run:
//   ./engine_probe [dll_path]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "include/LNTVulkanEngine.h"

using namespace lnt;

// ---- deterministic fill: a 20x20x20 lattice in a 60 m cube ---------------
static const double kOriginX = 682000.0, kOriginY = 3504000.0, kOriginZ = 1040.0;
static const int kSide = 20;
static const double kCube = 60.0;

struct Tile {
    uint64_t first = 0;
    uint64_t count = 0;
};

static std::vector<RawPoint> make_points() {
    std::vector<RawPoint> pts;
    pts.reserve(size_t(kSide) * kSide * kSide);
    for (int iz = 0; iz < kSide; ++iz)
        for (int iy = 0; iy < kSide; ++iy)
            for (int ix = 0; ix < kSide; ++ix) {
                RawPoint p{};
                const double t = (kSide > 1) ? double(kSide - 1) : 1.0;
                p.x = kOriginX + kCube * ix / t;
                p.y = kOriginY + kCube * iy / t;
                p.z = kOriginZ + kCube * iz / t;
                p.r = static_cast<uint8_t>(255 * ix / (kSide - 1));
                p.g = static_cast<uint8_t>(255 * iy / (kSide - 1));
                p.b = static_cast<uint8_t>(255 * iz / (kSide - 1));
                p.intensity = 200;
                pts.push_back(p);
            }
    return pts;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("LNT <-> VULKAN ENGINE PROBE\n");

    const std::string dll = (argc > 1) ? argv[1] : VulkanEngine::find_dll();
    std::printf("  dll              : %s\n",
                dll.empty() ? "(not found)" : dll.c_str());
    if (dll.empty()) {
        std::printf("RESULT: no engine DLL -> render path unavailable\n");
        return 2;
    }

    VulkanEngine eng;
    std::string err;
    if (!eng.load(dll, err)) {
        std::printf("  load             : FAILED - %s\n", err.c_str());
        std::printf("RESULT: engine not loadable\n");
        return 2;
    }
    std::printf("  load             : ok (%zu exports resolved)\n", size_t(33));
    std::printf("  is_available     : %d\n", eng.is_available() ? 1 : 0);
    if (!eng.is_available()) {
        std::printf("RESULT: no Vulkan loader / device\n");
        return 2;
    }

    // Hidden window on purpose: if pixel verification needed a visible window
    // this whole path would be untestable in a headless run.
    if (!eng.open(640, 480, /*hidden=*/true, /*validation=*/false, err)) {
        std::printf("  create_renderer  : FAILED - %s\n", err.c_str());
        std::printf("RESULT: device creation failed\n");
        return 2;
    }
    std::printf("  device           : %s\n", eng.device_name().c_str());
    std::printf("  vram             : %.2f GB (budget %.2f GB)  api=0x%08x\n",
                eng.vram_bytes() / 1073741824.0, eng.vram_budget() / 1073741824.0,
                eng.api_version());

    const std::vector<RawPoint> pts = make_points();
    const uint64_t n = pts.size();
    std::printf("  points           : %llu\n", (unsigned long long)n);

    // ---- render origin: the dataset centre, exactly as the .lnt header gives
    // it. At a 682000 m easting the float32 mantissa is ~6 cm, so leaving the
    // origin at 0 would throw away the payload precision this pipeline exists
    // to preserve.
    const double cx = (kOriginX + kOriginX + kCube) * 0.5;
    const double cy = (kOriginY + kOriginY + kCube) * 0.5;
    const double cz = (kOriginZ + kOriginZ + kCube) * 0.5;
    std::printf("  render origin    : (%.1f, %.1f, %.1f)\n", cx, cy, cz);
    eng.set_render_origin(cx, cy, cz);

    // ---- arena: two tiles so culling one is a real subset change -----------
    const uint64_t half = n / 2;
    const int r = eng.reserve_point_capacity(n);
    std::printf("  reserve %.0f pts : %d  (1=ready 2=reallocated)\n", double(n), r);
    if (r == 0) {
        std::printf("RESULT: arena reservation failed - %s\n", eng.last_error());
        return 1;
    }

    std::vector<double> xyz(n * 3);
    std::vector<uint8_t> cls(n);
    std::vector<float> inten(n);
    for (uint64_t i = 0; i < n; ++i) {
        xyz[i * 3 + 0] = pts[i].x;
        xyz[i * 3 + 1] = pts[i].y;
        xyz[i * 3 + 2] = pts[i].z;
        cls[i] = static_cast<uint8_t>(i % 8);
        inten[i] = pts[i].intensity;
    }
    const bool t0 = eng.upload_point_tile(0, half, xyz.data(), cls.data(), inten.data());
    const bool t1 = eng.upload_point_tile(half, n - half, xyz.data() + half * 3,
                                          cls.data() + half, inten.data() + half);
    std::printf("  upload tiles     : %d %d  (tiles=%llu capacity=%llu)\n",
                t0 ? 1 : 0, t1 ? 1 : 0,
                (unsigned long long)eng.arena_uploaded_tiles(),
                (unsigned long long)eng.point_capacity());
    if (!t0 || !t1) {
        std::printf("RESULT: tile upload failed - %s\n", eng.last_error());
        return 1;
    }

    eng.set_display_mode(kEngineRgb);
    eng.set_point_size_params(0.25f, 2.0f, 12.0f, 0.0f);
    // A distinctive background makes "did anything draw?" unambiguous instead
    // of a guess about what black means.
    eng.set_clear_color(1.0f, 0.0f, 0.0f, 1.0f);
    eng.set_camera_lookat(cx + 90.0, cy - 90.0, cz + 70.0, cx, cy, cz,
                          45.0, 0.5, 5000.0);

    const uint64_t uploads_before = eng.position_upload_count();

    // One measurement per draw-list state. Both numbers matter and they answer
    // different questions: draw-call DELTA says what the renderer was told to
    // draw, pixel count says what actually reached the framebuffer. A pixel
    // count alone cannot distinguish "culled to nothing" from "drawn but off
    // screen", so the harness needs both.
    uint64_t prev_calls = eng.point_draw_call_count();
    auto measure = [&](const char* label, bool use_ranges,
                       std::vector<uint32_t> firsts,
                       std::vector<uint32_t> counts) -> uint64_t {
        if (use_ranges)
            eng.set_point_draw_ranges(firsts, counts);
        else
            eng.clear_point_draw_ranges();
        const int rr = eng.render();
        EngineFrame f;
        if (!eng.capture_frame(f)) {
            std::printf("  %-14s capture FAILED: %s\n", label, eng.last_error());
            return ~0ull;
        }
        // Capture twice: a stale/lagged image would show up as the two frames
        // disagreeing, which would silently invalidate every pixel number.
        EngineFrame f2;
        const bool stable = eng.capture_frame(f2) && (f.rgba == f2.rgba);
        const uint64_t calls = eng.point_draw_call_count();
        const uint64_t px = f.count_not_equal(255, 0, 0, 8);
        std::printf("  %-14s ranges=%-2u pts=%-5u drawCalls=+%-3llu render=%d  "
                    "px=%-7llu stable=%d\n",
                    label, eng.draw_range_count(), eng.draw_range_points(),
                    (unsigned long long)(calls - prev_calls), rr,
                    (unsigned long long)px, stable ? 1 : 0);
        prev_calls = calls;
        // WHAT are those pixels? A count alone cannot tell a drawn point from a
        // format/clear-colour surprise, so print the dominant colours.
        std::unordered_map<uint32_t, uint64_t> hist;
        for (size_t i = 0; i + 3 < f.rgba.size(); i += 4) {
            const uint32_t k = (uint32_t(f.rgba[i]) << 16) |
                               (uint32_t(f.rgba[i + 1]) << 8) |
                               uint32_t(f.rgba[i + 2]);
            ++hist[k];
        }
        std::vector<std::pair<uint64_t, uint32_t>> top;   // (count, rgb)
        top.reserve(hist.size());
        for (const auto& kv : hist) top.emplace_back(kv.second, kv.first);
        std::sort(top.begin(), top.end(),
                  [](const std::pair<uint64_t, uint32_t>& a,
                     const std::pair<uint64_t, uint32_t>& b) { return a.first > b.first; });
        std::printf("                 top colours (count, rgb hex):");
        for (size_t i = 0; i < top.size() && i < 4; ++i)
            std::printf(" %llu=#%06x", (unsigned long long)top[i].first, top[i].second);
        std::printf("  [distinct=%zu of %zu px]\n", hist.size(),
                    size_t(f.width) * f.height);
        return px;
    };

    float cc[4] = {0, 0, 0, 0};
    eng.get_clear_color(cc);
    std::printf("  clear colour read back: %.3f %.3f %.3f %.3f\n",
                cc[0], cc[1], cc[2], cc[3]);

    std::printf("\n  draw-list states:\n");
    // Empty FIRST, before any range was ever set: this separates "empty list"
    // from "a previously-set list was not cleared".
    const uint64_t px_empty_first = measure("empty(first)", false, {}, {});
    const uint64_t px_all = measure("all tiles", true,
                                    {0u, uint32_t(half)},
                                    {uint32_t(half), uint32_t(n - half)});
    const uint64_t px_half = measure("tile0 only", true, {0u}, {uint32_t(half)});
    const uint64_t px_empty_after = measure("empty(after)", false, {}, {});
    const uint64_t px_all_again = measure("all again", true,
                                          {0u, uint32_t(half)},
                                          {uint32_t(half), uint32_t(n - half)});

    const uint64_t uploads_after = eng.position_upload_count();
    std::printf("  position uploads : %llu -> %llu (draw-range changes must not move it)\n",
                (unsigned long long)uploads_before, (unsigned long long)uploads_after);
    std::printf("  gpu frame time   : %.3f ms\n", eng.gpu_frame_time_ms());

    std::printf("\nVERDICT\n");
    const bool a = px_all > 0;
    const bool b = px_empty_first == 0;
    const bool c = px_half > 0 && px_half < px_all;
    const bool d = px_empty_after == 0;
    const bool e = px_all_again == px_all;
    const bool f = uploads_before == uploads_after;
    const bool g = eng.point_draw_call_count() > 0;
    std::printf("  draws pixels               : %s\n", a ? "YES" : "NO");
    std::printf("  empty list draws nothing   : %s", b ? "YES" : "NO");
    if (!b) std::printf("  (%llu px)", (unsigned long long)px_empty_first);
    std::printf("\n  subset is strictly smaller : %s\n", c ? "YES" : "NO");
    std::printf("  clearing again empties it  : %s\n", d ? "YES" : "NO");
    std::printf("  full set reproducible      : %s\n", e ? "YES" : "NO");
    std::printf("  no re-upload on range churn: %s\n", f ? "YES" : "NO");
    std::printf("  draw calls actually issued : %s\n", g ? "YES" : "NO");

    eng.close();
    const bool good = a && b && c && d && e && f && g;
    std::printf("RESULT: %s\n", good ? "engine drives .lnt points end to end" : "FAILED");
    return good ? 0 : 1;
}
