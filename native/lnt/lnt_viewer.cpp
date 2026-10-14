// lnt_viewer.cpp - standalone interactive viewer for .lnt (and .las, converted on
// the fly). The thin shell around the pieces the test harness already verifies:
//   LASReader -> LNTBuilder -> Reader -> LntStreamer/LntCacheManager ->
//   LntInteractiveLoop -> VulkanEngine.
//
//   lnt_viewer <file.las|file.lnt> [--sse N] [--budget-mb N] [--size WxH]
//              [--lnt-out path.lnt] [--screenshot out.png [--frames N]]
//
// Controls: left-drag orbit | right/middle-drag pan | wheel zoom |
//           [ / ]  or  Down / Up  SSE threshold -/+ (0 = full resolution, max 50) |
//           1 elevation, 2 intensity, 3 neutral | R reset view | Esc quit.
//
// The initial view is PLAN view (looking down -Z, X right, Y up), which in the
// engine's Y-up camera is yaw 0 / pitch 0.
//
// Build (from native/lnt):
//   g++ -O3 -std=c++17 -static -o lnt_viewer.exe lnt_viewer.cpp LNTReader.cpp \
//       src/LNTStreamer.cpp src/LNTLASReader.cpp src/LNTCameraController.cpp \
//       src/LNTInteractiveLoop.cpp src/LNTBuilder.cpp -DLNT_EXTERN_BUILDER

#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "include/LNTBuilder.h"
#include "include/LNTCacheManager.h"
#include "include/LNTCameraController.h"
#include "include/LNTInteractiveLoop.h"
#include "include/LNTLASReader.h"
#include "include/LNTReader.h"
#include "include/LNTRenderBridge.h"
#include "include/LNTStreamer.h"
#include "include/LNTVulkanEngine.h"

using namespace lnt;

namespace {

bool ends_with_ci(const std::string& s, const char* suf) {
    const size_t n = std::strlen(suf);
    if (s.size() < n) return false;
    for (size_t i = 0; i < n; ++i)
        if (std::tolower((unsigned char)s[s.size() - n + i]) != suf[i]) return false;
    return true;
}

bool file_newer(const std::string& a, const std::string& b) {   // a newer than b
    WIN32_FILE_ATTRIBUTE_DATA da, db;
    if (!GetFileAttributesExA(a.c_str(), GetFileExInfoStandard, &da)) return false;
    if (!GetFileAttributesExA(b.c_str(), GetFileExInfoStandard, &db)) return true;
    return CompareFileTime(&da.ftLastWriteTime, &db.ftLastWriteTime) > 0;
}

// ---- minimal PNG writer (stored deflate blocks) --------------------------
uint32_t crc32_of(const uint8_t* p, size_t n, uint32_t crc = 0xFFFFFFFFu) {
    static uint32_t t[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[i] = c;
        }
        init = true;
    }
    for (size_t i = 0; i < n; ++i) crc = t[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return crc;
}
void put32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(uint8_t(x >> 24)); v.push_back(uint8_t(x >> 16));
    v.push_back(uint8_t(x >> 8));  v.push_back(uint8_t(x));
}
void png_chunk(std::vector<uint8_t>& out, const char* type, const std::vector<uint8_t>& data) {
    put32(out, uint32_t(data.size()));
    std::vector<uint8_t> td(type, type + 4);
    td.insert(td.end(), data.begin(), data.end());
    out.insert(out.end(), td.begin(), td.end());
    put32(out, crc32_of(td.data(), td.size()) ^ 0xFFFFFFFFu);
}
bool write_png(const std::string& path, const EngineFrame& f) {
    std::vector<uint8_t> raw;
    raw.reserve(size_t(f.height) * (size_t(f.width) * 3 + 1));
    for (uint32_t y = 0; y < f.height; ++y) {
        raw.push_back(0);
        for (uint32_t x = 0; x < f.width; ++x) {
            const uint8_t* p = f.rgba.data() + (size_t(y) * f.width + x) * 4;
            raw.push_back(p[0]); raw.push_back(p[1]); raw.push_back(p[2]);
        }
    }
    std::vector<uint8_t> z = {0x78, 0x01};
    uint32_t a = 1, b = 0;
    for (uint8_t c : raw) { a = (a + c) % 65521; b = (b + a) % 65521; }
    for (size_t pos = 0; pos < raw.size();) {
        const size_t n = std::min<size_t>(65535, raw.size() - pos);
        z.push_back(pos + n >= raw.size() ? 1 : 0);
        z.push_back(uint8_t(n)); z.push_back(uint8_t(n >> 8));
        z.push_back(uint8_t(~n)); z.push_back(uint8_t((~n) >> 8));
        z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + n);
        pos += n;
    }
    put32(z, (b << 16) | a);
    std::vector<uint8_t> out = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::vector<uint8_t> ihdr;
    put32(ihdr, f.width); put32(ihdr, f.height);
    ihdr.push_back(8); ihdr.push_back(2); ihdr.push_back(0); ihdr.push_back(0); ihdr.push_back(0);
    png_chunk(out, "IHDR", ihdr);
    png_chunk(out, "IDAT", z);
    png_chunk(out, "IEND", {});
    std::ofstream o(path, std::ios::binary);
    o.write(reinterpret_cast<const char*>(out.data()), std::streamsize(out.size()));
    return bool(o);
}

void usage() {
    std::printf("usage: lnt_viewer <file.las|file.lnt> [--sse N] [--budget-mb N] [--size WxH]\n"
                "                  [--screenshot out.png [--frames N]]\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string input, shot, lnt_out;
    double sse = 8.0;
    uint64_t budget_mb = 512;
    uint32_t W = 1280, H = 720;
    int shot_frames = 150;
    int shot_mode = kEngineElevation;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--sse") sse = std::atof(next());
        else if (a == "--budget-mb") budget_mb = std::strtoull(next(), nullptr, 10);
        else if (a == "--size") std::sscanf(next(), "%ux%u", &W, &H);
        else if (a == "--screenshot") shot = next();
        else if (a == "--frames") shot_frames = std::atoi(next());
        else if (a == "--mode") shot_mode = std::atoi(next());
        else if (a == "--lnt-out") lnt_out = next();
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (input.empty()) input = a;
    }
    if (input.empty()) { usage(); return 2; }

    // ---- 1. LAS -> .lnt when needed --------------------------------------
    std::string lnt_path = input;
    if (ends_with_ci(input, ".las")) {
        lnt_path = lnt_out.empty() ? input.substr(0, input.size() - 4) + ".lnt" : lnt_out;
        if (file_newer(input, lnt_path)) {
            std::printf("[convert] %s -> %s\n", input.c_str(), lnt_path.c_str());
            LNTBuildParams params;
            LNTBuildStats st;
            std::string err;
            const auto t0 = std::chrono::steady_clock::now();
            if (!convert_las_to_lnt(input, lnt_path, params, &st, &err)) {
                std::printf("[convert] FAILED: %s\n", err.c_str());
                return 1;
            }
            std::printf("[convert] %llu points, %llu nodes, depth %llu, %.1f MiB, %.1f s\n",
                        (unsigned long long)st.input_points, (unsigned long long)st.node_count,
                        (unsigned long long)st.max_depth, double(st.file_bytes) / (1024 * 1024),
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        } else {
            std::printf("[convert] %s is up to date, reusing it\n", lnt_path.c_str());
        }
    } else if (ends_with_ci(input, ".laz")) {
        std::printf("LAZ is not supported by this viewer. Convert it to LAS first "
                    "(e.g. laszip -i in.laz -o out.las).\n");
        return 2;
    } else if (!ends_with_ci(input, ".lnt")) {
        std::printf("unsupported extension (expected .las or .lnt): %s\n", input.c_str());
        return 2;
    }

    // ---- 2. open the file --------------------------------------------------
    std::string err;
    Reader rd;
    if (!rd.open(lnt_path)) { std::printf("cannot open %s: %s\n", lnt_path.c_str(), rd.error().c_str()); return 1; }
    const BoundingBox env = rd.get_bounds();
    const double side = env.extent(0);
    double origin[3];
    for (int i = 0; i < 3; ++i) origin[i] = rd.origin(i);
    std::printf("[file] %s: %llu points, %llu nodes, extent %.1f x %.1f x %.1f m\n", lnt_path.c_str(),
                (unsigned long long)rd.get_total_points(), (unsigned long long)rd.get_total_nodes(),
                env.extent(0), env.extent(1), env.extent(2));

    // ---- 3. engine ---------------------------------------------------------
    VulkanEngine eng;
    const std::string dll = VulkanEngine::find_dll();
    if (dll.empty()) { std::printf("naksha_vulkan.dll not found (set LNT_VULKAN_DLL)\n"); return 1; }
    if (!eng.load(dll, err)) { std::printf("engine load failed: %s\n", err.c_str()); return 1; }
    if (!eng.is_available()) { std::printf("no Vulkan device\n"); return 1; }
    const bool headless = !shot.empty();
    if (!eng.open(W, H, headless, false, err)) { std::printf("renderer failed: %s\n", err.c_str()); return 1; }

    LntScene scene;
    if (!scene.plan(rd, err) || !scene.reserve(eng, err)) {
        std::printf("arena reservation failed (%llu points): %s\n",
                    (unsigned long long)rd.get_total_points(), err.c_str());
        return 1;
    }
    eng.set_render_origin(origin[0], origin[1], origin[2]);
    eng.set_clear_color(0.0f, 0.0f, 0.0f, 1.0f);
    eng.set_display_mode(kEngineClassification);
    eng.set_point_size_params(0.12f, 1.5f, 8.0f, 0.0f);

    LntCacheManager cache(budget_mb * 1024ull * 1024ull);
    LntStreamer streamer;
    if (!cache.open(lnt_path, err) || !streamer.attach(rd, &cache, err)) {
        std::printf("streamer failed: %s\n", err.c_str());
        return 1;
    }

    LntCameraController ctl;
    LntInteractiveLoop loop;
    auto reset_view = [&]() {
        ctl.set_target(env.mid(0), env.mid(1), env.mid(2));
        ctl.set_lens(45.0, 1.0, std::max(side, 1.0) * 40.0);
        ctl.set_ortho2d(true);                          // 2D plan view (orthographic)
        ctl.set_distance(std::max(side, 1.0) * 1.3);
        ctl.set_sse_threshold(sse);
    };
    reset_view();
    loop.attach(&ctl, &cache, origin);
    loop.set_viewport(W, H);

    // ---- headless self-check ----------------------------------------------
    if (headless) {
        eng.set_display_mode(shot_mode);
        for (int f = 0; f < shot_frames; ++f) {
            if (!loop.render_frame(eng, scene, streamer, 1.0f / 60.0f)) {
                std::printf("frame %d failed: %s\n", f, loop.error().c_str());
                return 1;
            }
            Sleep(4);
        }
        EngineFrame fr;
        const LNTStreamerStats& s = loop.stats();
        if (!loop.render_frame(eng, scene, streamer, 0.0f) || !eng.capture_frame(fr)) {
            std::printf("capture failed\n");
            return 1;
        }
        uint64_t lit = 0;
        // Background = the corner pixel (the framebuffer is sRGB, so the clear
        // colour is not the linear value that was set).
        const uint8_t* bg = fr.rgba.data();
        for (size_t i = 0; i + 3 < fr.rgba.size(); i += 4)
            if (std::abs(int(fr.rgba[i]) - bg[0]) > 6 || std::abs(int(fr.rgba[i + 1]) - bg[1]) > 6 ||
                std::abs(int(fr.rgba[i + 2]) - bg[2]) > 6) ++lit;
        std::printf("[shot] %u nodes, %u points drawn (of %u wanted), cache %.1f/%.1f MiB, "
                    "%llu non-background pixels of %zu\n",
                    s.active_nodes, s.total_points_rendered, s.wanted_points,
                    double(s.cache_bytes_used) / (1024 * 1024),
                    double(s.cache_capacity_bytes) / (1024 * 1024), (unsigned long long)lit,
                    fr.rgba.size() / 4);
        if (!write_png(shot, fr)) { std::printf("cannot write %s\n", shot.c_str()); return 1; }
        std::printf("[shot] wrote %s\n", shot.c_str());
        eng.close();
        cache.close();
        return lit > 0 ? 0 : 3;
    }

    // ---- interactive -------------------------------------------------------
    HWND hwnd = eng.hwnd();
    ShowWindow(hwnd, SW_SHOWNORMAL);
    SetForegroundWindow(hwnd);
    bool running = true, dragging_l = false, dragging_p = false;
    int last_x = 0, last_y = 0;
    auto last_t = std::chrono::steady_clock::now();
    auto title_t = last_t;
    int frames = 0;
    double fps = 0.0;
    int mode = kEngineElevation;
    std::printf("[viewer] controls: drag=orbit, right-drag=pan, wheel=zoom, [ ]=SSE, 1/2/3=colour, R=reset, Esc=quit\n");

    while (running && IsWindow(hwnd)) {
        MSG m;
        while (PeekMessageA(&m, nullptr, 0, 0, PM_REMOVE)) {
            if (m.message == WM_QUIT) running = false;
            if (m.hwnd == hwnd) {
                const int x = int(short(LOWORD(m.lParam))), y = int(short(HIWORD(m.lParam)));
                switch (m.message) {
                case WM_LBUTTONDOWN: dragging_l = true; last_x = x; last_y = y; SetCapture(hwnd); break;
                case WM_LBUTTONUP:   dragging_l = false; if (!dragging_p) ReleaseCapture(); break;
                case WM_RBUTTONDOWN:
                case WM_MBUTTONDOWN: dragging_p = true; last_x = x; last_y = y; SetCapture(hwnd); break;
                case WM_RBUTTONUP:
                case WM_MBUTTONUP:   dragging_p = false; if (!dragging_l) ReleaseCapture(); break;
                case WM_MOUSEMOVE:
                    if (dragging_l)
                        ctl.orbit(float(-(x - last_x) * 0.005), float((y - last_y) * 0.005));
                    if (dragging_p) {
                        const float k = float(ctl.distance() * 0.0012);
                        ctl.pan(float(-(x - last_x)) * k, float(y - last_y) * k);
                    }
                    last_x = x; last_y = y;
                    break;
                case WM_MOUSEWHEEL: {
                    const int d = GET_WHEEL_DELTA_WPARAM(m.wParam);
                    ctl.zoom(float(-ctl.goal_distance() * 0.12 * (d / 120.0)));
                    break;
                }
                case WM_KEYDOWN:
                    switch (m.wParam) {
                    case VK_ESCAPE: running = false; break;
                    case VK_UP: case VK_OEM_6: ctl.adjust_sse(+1.0); break;      // ]
                    case VK_DOWN: case VK_OEM_4: ctl.adjust_sse(-1.0); break;    // [
                    case 'R': reset_view(); break;
                    case 'D': ctl.set_ortho2d(!ctl.ortho2d()); break;   // 2D <-> 3D orbit
                    case '1': mode = kEngineElevation; eng.set_display_mode(mode); break;
                    case '2': mode = kEngineIntensity; eng.set_display_mode(mode); break;
                    case '3': mode = kEngineNeutral; eng.set_display_mode(mode); break;
                    default: break;
                    }
                    break;
                default: break;
                }
            }
            TranslateMessage(&m);
            DispatchMessageA(&m);
        }
        if (!running || !IsWindow(hwnd)) break;

        RECT rc;
        GetClientRect(hwnd, &rc);
        const uint32_t cw = uint32_t(rc.right - rc.left), ch = uint32_t(rc.bottom - rc.top);
        if (cw > 0 && ch > 0 && (cw != W || ch != H)) {
            W = cw; H = ch;
            eng.resize(W, H);
            loop.set_viewport(W, H);
        }
        if (IsIconic(hwnd)) { Sleep(30); continue; }

        const auto now = std::chrono::steady_clock::now();
        const float dt = float(std::chrono::duration<double>(now - last_t).count());
        last_t = now;
        if (!loop.render_frame(eng, scene, streamer, dt)) {
            std::printf("[viewer] frame failed: %s\n", loop.error().c_str());
            break;
        }
        ++frames;
        const double since = std::chrono::duration<double>(now - title_t).count();
        if (since >= 0.5) {
            fps = frames / since;
            frames = 0;
            title_t = now;
            const LNTStreamerStats& s = loop.stats();
            char t[256];
            std::snprintf(t, sizeof(t),
                          "LNT viewer | %.0f fps | nodes %u | points %u/%u | cache %.0f/%.0f MiB | "
                          "loading %u | SSE %.0f px | dist %.0f m",
                          fps, s.active_nodes, s.total_points_rendered, s.wanted_points,
                          double(s.cache_bytes_used) / (1024 * 1024),
                          double(s.cache_capacity_bytes) / (1024 * 1024), s.pending_async_loads,
                          s.sse_threshold, ctl.distance());
            SetWindowTextA(hwnd, t);
        }
        Sleep(1);
    }
    eng.close();
    cache.close();
    return 0;
}
