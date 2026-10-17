// LNTVulkanEngine.h - drive the existing naksha_vulkan engine from the native
// .lnt pipeline.
//
// WHY A RUNTIME-LOADED C ABI. The .lnt converter builds with MinGW g++
// (`g++ -O3 -std=c++17 ... -o main`, no other dependency), while
// naksha_vulkan.dll is produced by MSVC + CMake + Ninja against the Vulkan SDK.
// MinGW cannot consume the MSVC import library (naksha_vulkan.lib), so linking
// is not available even in principle. The engine already publishes a flat
// extern "C" ABI for exactly this reason - it is driven from Python via ctypes
// today - so this header resolves those same exports with LoadLibrary /
// GetProcAddress. That keeps zero build coupling between the two toolchains and
// no new dependency in the .lnt build command.
//
// NOTHING VULKAN-SPECIFIC CROSSES THIS BOUNDARY either: the only handle is the
// engine's own opaque uint64, and no <vulkan/*.h> is included here. So the .lnt
// pipeline compiles on a machine with no Vulkan SDK installed at all; it simply
// reports the engine as unavailable and the render checks are skipped.
//
// THE ARENA IS THE POINT. `nkv_set_point_cloud` re-packs the whole cloud at
// offset 0, so adding or hiding one node would re-upload every point. The
// stable-offset arena (nkv_reserve_point_capacity + nkv_upload_point_tile) lets
// each .lnt node be written once at a caller-chosen offset, and
// nkv_set_point_draw_ranges then selects what is drawn - which is exactly the
// shape of a node-based format: one LNTNodeEntry is one tile, and node-level
// culling becomes a draw-range update with no buffer traffic at all.

#ifndef LNT_VULKAN_ENGINE_H
#define LNT_VULKAN_ENGINE_H

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "LNTTypes.h"
#include "LNTClassificationTable.h"
#include "LNTShadingModes.h"
#include "LNTSurfaceMesh.h"

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

namespace lnt {

// ---- engine display modes (mirrors NkvDisplayMode in the C API) ---------
enum EngineDisplayMode : int {
    kEngineRgb = 0,
    kEngineClassification = 1,
    kEngineIntensity = 2,
    kEngineElevation = 3,
    kEngineShadedClassInstant = 4,
    kEngineNeutral = 5,
    kEngineDepth = 6,
};

// ---- an rgba8 image captured offscreen from the engine -----------------
struct EngineFrame {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> rgba;      // width * height * 4, rows top-to-bottom

    // Pixel comparison helpers. The clear colour is known, so "is this pixel
    // background?" is a real question with a real answer rather than a guess
    // about what black means.
    uint64_t count_not_equal(uint8_t r, uint8_t g, uint8_t b, int tol = 0) const;
    uint64_t count_where(bool (*pred)(uint8_t, uint8_t, uint8_t), int* dummy = nullptr) const;
    const uint8_t* at(uint32_t x, uint32_t y) const;
};

// ========================================================================
// VulkanEngine
// ========================================================================
// Lifecycle is explicit and never throws: a missing DLL, a machine without a
// Vulkan device, or a failed device creation all leave the object in a
// reportable failed state (ok() == false, error() explains) rather than
// crashing the converter or the test harness.
class VulkanEngine {
public:
    VulkanEngine() = default;
    ~VulkanEngine();
    VulkanEngine(const VulkanEngine&) = delete;
    VulkanEngine& operator=(const VulkanEngine&) = delete;

    // Candidate DLL locations, most specific first. An LNT_VULKAN_DLL
    // environment variable wins outright so a test can point at a specific
    // build. Returns an empty string when nothing is found.
    static std::string find_dll();
    static std::vector<std::string> dll_candidates();

    // Steps 1+2: load the library and resolve the ABI. No device is created.
    bool load(const std::string& dll_path, std::string& err);
    bool loaded() const { return ready_; }

    // True when the loader is present AND at least one physical device exists.
    // Side-effect free, safe to call before open().
    bool is_available() const;

    // Steps 3+4: create the WIN32 window (visible or hidden) and the renderer's
    // device/swapchain bound to it. `hidden` is what makes headless testing
    // possible: the engine's offscreen capture does not need a presented frame.
    // `parent` (optional, a Win32 HWND passed as void*) makes the host window a
    // visible WS_CHILD of that window - how a Qt widget embeds the renderer.
    bool open(uint32_t width, uint32_t height, bool hidden, bool validation,
              std::string& err, void* parent = nullptr);
    // Binds to a renderer ANOTHER component already created (the engine allows
    // one renderer per process, and Naksha owns it). Requires load(). The handle
    // is borrowed: close() detaches without nkv_destroy_renderer and without
    // touching any window.
    bool attach_existing(uint64_t handle);
    // Creates the renderer on a window the CALLER owns (a Qt widget's HWND). The
    // engine owns the renderer (close() destroys it) but never the window.
    bool open_on_window(void* hwnd, uint32_t width, uint32_t height, std::string& err);
    uint64_t raw_handle() const { return handle_; }   // for hosts that share the renderer
    void close();

    bool ok() const { return handle_ != 0; }
#if defined(_WIN32)
    // The host window the swapchain is bound to (viewer: show it, pump its input).
    HWND hwnd() const { return hwnd_; }
#endif
    const std::string& error() const { return error_; }
    const std::string& dll_path() const { return dll_path_; }

    // ---- device introspection -------------------------------------------
    std::string device_name() const;
    uint64_t vram_bytes() const;
    uint64_t vram_budget() const;
    uint32_t api_version() const;

    // ---- arena ----------------------------------------------------------
    // 0 = failed, 1 = ready, 2 = (re)allocated (every earlier tile invalid).
    int reserve_point_capacity(uint64_t capacity_points);
    // Positions are WORLD float64; the engine converts against the render
    // origin. Requirement: point index `first + i` of this tile is the same
    // canonical point in every stream, so a node's points must be packed
    // contiguously and one tile must not overlap another.
    // rgb8 (optional, count*3 bytes) is the true-colour stream of the RGB display mode; it
    // needs an engine with nkv_upload_point_tile_ex (see has_rgb_tiles()).
    bool upload_point_tile(uint64_t first, uint64_t count,
                           const double* xyz_world_f64,
                           const uint8_t* classification,
                           const float* intensity,
                           const uint8_t* rgb8 = nullptr);
    bool has_rgb_tiles() const { return api_.upload_point_tile_ex != nullptr; }
    // Presentation mode (RGB / Intensity / Elevation / Classification / NormalShading):
    // one uniform update - the display mode, the two ramp tables, the elevation and
    // intensity ranges, the light and the Eye-Dome Lighting strength. No vertex buffer is
    // touched. False when the engine DLL predates nkv_set_shading_mode.
    bool set_shading_mode(const ShadingUniforms& config);
    bool has_shading_mode() const { return api_.set_shading_mode != nullptr; }

    // ---- shaded surface (triangulated hillshade) ---------------------------------------
    // Uploads a mesh built by build_surface_mesh() through the engine's tiled-surface path (it
    // is promoted to the visible set atomically at the next frame boundary, so there is no
    // blank frame). `revision` must increase with every upload (stale-set guard).
    bool upload_surface(const SurfaceMesh& mesh, uint64_t revision);
    bool has_vertex_surface() const { return api_.set_surface_indexed_blocks_vc != nullptr; }
    // ---- crisp shaded-classification surface (VTK parity) ------------------------------
    // The engine's surface.frag modes 4/5 are a transcription of gui/shading_display.py's
    // Shaded Classification model (crisp flat facet base + barycentric mixed-class overlay,
    // sharpness-overdrive light elevation, key/fill rig). Uploading the TIN through this
    // path gives the .lnt view the SAME shading the VTK view has, because the engine derives
    // the hillshade in-shader from the same parameters instead of the CPU-baked `colors`
    // the legacy path writes. `class_color_lut` is 256*3 RGB. `revision` must increase.
    bool upload_shaded_class_surface(const SurfaceMesh& mesh, const uint8_t* class_color_lut,
                                     uint64_t revision);
    bool has_shaded_class_surface() const {
        return (api_.set_shaded_class_surface != nullptr ||
                api_.set_shaded_class_surface_indexed != nullptr) &&
               api_.set_crisp_shading_parameters != nullptr;
    }
    // The compact crisp path draws the per-vertex class colours itself: the mesher need
    // not bake per-face colours for it.
    bool has_compact_shaded_class_surface() const {
        return api_.set_shaded_class_surface_indexed != nullptr &&
               api_.set_crisp_shading_parameters != nullptr;
    }
    // Push-constant only: never re-uploads geometry. Mirrors
    // _configure_microstation_color_blend_lighting()'s light rig.
    bool has_surface_elevation_color() const { return api_.surface_elevation_color != nullptr; }
    bool set_surface_elevation_color(bool on, double z0, double z1, uint32_t ramp) {
        return handle_ && api_.surface_elevation_color && api_.surface_elevation_color(handle_, on, float(z0), float(z1), float(ramp));
    }
    bool set_crisp_shading_parameters(double azimuth_deg, double sharpness_raw, double ambient,
                                      double key_intensity, double fill_intensity);
    // ---- viewport-LOD surface tiles (engine tile arena; LNTSurfaceLod.h) -------------
    // Mirrors NkvSurfaceTileDesc / NkvSurfaceTileDraw (naksha_vulkan_c_api.h) byte for byte.
    struct SurfaceTileUpload {
        uint64_t key;
        const double* positions;
        const uint8_t* vertex_class;
        const int32_t* faces;              // local vertex indices
        const double* coverage_positions;
        const uint8_t* coverage_class;
        uint32_t vertex_count, face_count, coverage_count;
        int32_t result;                    // out: 1 resident, 0 rejected, -1 arena full
    };
    struct SurfaceTileDraw {
        uint64_t key;
        uint32_t first_face, face_count;   // tile-local triangle range
        uint32_t coverage, _pad;           // 1 = also draw the tile's coverage points
    };
    bool has_surface_tiles() const {
        return api_.surface_tiles_configure && api_.surface_tiles_upload && api_.surface_tiles_evict &&
               api_.surface_tiles_set_draw_list && api_.surface_tiles_set_palette &&
               api_.surface_tiles_active && api_.surface_tiles_release && api_.set_crisp_shading_parameters;
    }
    bool surface_tiles_configure(uint64_t vertices, uint64_t faces, uint64_t coverage) {
        return handle_ && api_.surface_tiles_configure &&
               api_.surface_tiles_configure(handle_, vertices, faces, coverage) != 0;
    }
    bool surface_tiles_release() {
        return handle_ && api_.surface_tiles_release && api_.surface_tiles_release(handle_) != 0;
    }
    bool surface_tiles_active() const {
        return handle_ && api_.surface_tiles_active && api_.surface_tiles_active(handle_) != 0;
    }
    int surface_tiles_upload(std::vector<SurfaceTileUpload>& tiles) {
        if (!handle_ || !api_.surface_tiles_upload || tiles.empty()) return 0;
        return api_.surface_tiles_upload(handle_, tiles.data(), uint32_t(tiles.size()));
    }
    bool surface_tiles_evict(const std::vector<uint64_t>& keys) {
        return handle_ && api_.surface_tiles_evict && !keys.empty() &&
               api_.surface_tiles_evict(handle_, keys.data(), uint32_t(keys.size())) != 0;
    }
    bool surface_tiles_set_palette(const uint8_t* lut_rgb, float coverage_shade) {
        return handle_ && api_.surface_tiles_set_palette &&
               api_.surface_tiles_set_palette(handle_, lut_rgb, coverage_shade) != 0;
    }
    bool surface_tiles_patch_classes(uint64_t key, const std::vector<uint32_t>& vi, const std::vector<uint8_t>& vc,
                                     const std::vector<uint32_t>& ci, const std::vector<uint8_t>& cc) {
        return handle_ && api_.surface_tiles_patch_classes &&
               api_.surface_tiles_patch_classes(handle_, key, vi.data(), vc.data(), uint32_t(vi.size()),
                                                ci.data(), cc.data(), uint32_t(ci.size())) != 0;
    }
    bool surface_tiles_set_draw_list(const std::vector<SurfaceTileDraw>& items) {
        return handle_ && api_.surface_tiles_set_draw_list &&
               api_.surface_tiles_set_draw_list(handle_, items.empty() ? nullptr : items.data(),
                                                uint32_t(items.size())) != 0;
    }
    bool surface_tiles_stats(uint64_t* out, uint32_t count) const {
        return handle_ && api_.surface_tiles_get_stats && api_.surface_tiles_get_stats(handle_, out, count) != 0;
    }
    // false = the streamed points are drawn, true = the surface is drawn instead
    bool show_surface(bool on);
    bool set_surface_lod_error_pixels(float pixels);
    uint32_t surface_lod() const;
    uint64_t surface_triangles() const;
    int surface_state() const;     // 0 empty, 1 preview, 2 final pending, 3 final active
    bool has_surface() const {
        return api_.set_surface_tiled != nullptr && api_.set_streaming_surface_active != nullptr;
    }
    uint64_t point_capacity() const;
    uint64_t arena_uploaded_tiles() const;

    // ---- draw ranges (node-level culling) --------------------------------
    // (first, count) windows into the already-resident arena. Never re-uploads.
    bool set_point_draw_ranges(const std::vector<uint32_t>& firsts,
                               const std::vector<uint32_t>& counts);
    bool clear_point_draw_ranges();
    uint32_t draw_range_count() const;
    uint32_t draw_range_points() const;

    // ---- point cloud (bulk, convenience) ---------------------------------
    bool set_point_cloud(const double* xyz, const uint8_t* rgb,
                         const uint8_t* classification, const float* intensity,
                         uint64_t count);
    bool set_point_cloud_visible(bool visible);
    bool set_display_mode(int mode);
    bool set_point_size_params(float footprint_m, float min_px, float max_px,
                               float class_intensity_mix);

    // ---- camera / frame --------------------------------------------------
    // WORLD coordinates: the engine subtracts the render origin internally.
    // Orthographic (VTK "parallel") camera: centre + view direction + parallel
    // scale (half the visible height in world units). Naksha's own 2D plan mode.
    bool set_camera_ortho(double cx, double cy, double cz, double dx, double dy, double dz,
                          double parallel_scale, double aspect, double near_clip,
                          double far_clip);
    // 256 x RGB (uint8) colour tables read by the point shader: classification
    // palette, elevation ramp, intensity ramp. nullptr leaves a table unchanged.
    bool set_point_luts(const uint8_t* class_rgb, const uint8_t* elevation_rgb,
                        const uint8_t* intensity_rgb);
    bool set_classification_lut(const PackedClassColor lut[256]);
    // Cross-section clip volume: up to kMaxClipPlanes planes (A,B,C,D) in WORLD
    // coordinates, normal pointing into the KEPT side. The GPU discards every point
    // on the negative side in the vertex stage - no CPU buffer is touched, so the
    // section appears and disappears with one tiny uniform update. nullptr / count 0
    // switches clipping off. False when the engine DLL predates nkv_set_clip_volume.
    bool set_clip_volume(const float* planes_abcd, uint32_t plane_count);
    bool has_clip_volume() const { return api_.set_clip_volume != nullptr; }
    // Same, from float64 WORLD planes. The engine's planes live in RENDER space
    // (world - render origin) because a float32 D at a UTM-scale coordinate (~1e6 m)
    // has a 6 cm ulp, which would move the cut by centimetres; the shift is applied
    // here in double: D_render = D + n . origin. Uses the origin last set through
    // set_render_origin().
    bool set_clip_planes_world(const double* abcd, uint32_t plane_count);
    bool set_camera_lookat(double ex, double ey, double ez,
                           double tx, double ty, double tz,
                           double fov_y_degrees, double near_clip, double far_clip);
    bool set_camera_view_up(double x, double y, double z);
    bool set_render_origin(double x, double y, double z);
    bool resize(uint32_t width, uint32_t height);
    // 2 = drawn+submitted+presented, 1 = skipped (minimized/out-of-date, NOT an
    // error), 0 = failure. A caller that needs "a real frame reached a screen"
    // must test for 2 - see the engine's own contract.
    int render();
    bool set_clear_color(float r, float g, float b, float a);
    bool get_clear_color(float out_rgba[4]) const;

    // Offscreen readback of the CURRENT camera view. This is the only honest
    // way to read engine pixels (GDI/PrintWindow see nothing of a Vulkan
    // surface), and being offscreen it works with a hidden window and no
    // presented frame at all. BLOCKING - test/parity path, never per frame.
    bool capture_frame(EngineFrame& out);
    bool get_last_mvp(float out_mvp16[16]);

    // ---- counters --------------------------------------------------------
    uint64_t point_count() const;
    uint64_t position_upload_count() const;
    uint64_t point_draw_call_count() const;
    double gpu_frame_time_ms() const;
    bool frame_stats(uint64_t* rendered, uint64_t* skipped,
                     uint64_t* recreates) const;
    const char* last_error() const;

private:
    // ---- resolved C ABI -------------------------------------------------
    // Every pointer is null until load() resolves it. A null checks the
    // operation as failed rather than calling through garbage.
    struct Api {
        bool ready = false;
#if defined(_WIN32)
        HMODULE lib = nullptr;
#endif
        int          (*is_available)(void) = nullptr;
        uint64_t     (*create_renderer)(void*, uint32_t, uint32_t, int) = nullptr;
        void         (*destroy_renderer)(uint64_t) = nullptr;
        void         (*clear)(uint64_t) = nullptr;
        int          (*set_point_cloud)(uint64_t, const double*, const uint8_t*,
                                        const uint8_t*, const float*, uint64_t) = nullptr;
        int          (*set_point_cloud_visible)(uint64_t, int) = nullptr;
        int          (*set_display_mode)(uint64_t, int) = nullptr;
        int          (*set_point_size_params)(uint64_t, float, float, float, float) = nullptr;
        int          (*reserve_point_capacity)(uint64_t, uint64_t) = nullptr;
        int          (*upload_point_tile)(uint64_t, uint64_t, uint64_t,
                                          const double*, const uint8_t*,
                                          const float*, const uint8_t*) = nullptr;
        int          (*upload_point_tile_ex)(uint64_t, uint64_t, uint64_t, const double*,
                                             const uint8_t*, const float*, const uint8_t*,
                                             const uint8_t*) = nullptr;
        int          (*set_shading_mode)(uint64_t, const ShadingUniforms*) = nullptr;
        int          (*set_surface_tiled)(uint64_t, const double*, uint64_t, const int32_t*, uint64_t,
                                          const uint8_t*, uint32_t, float, int, uint64_t,
                                          uint64_t) = nullptr;
        // Indexed upload: shared vertices + one colour per triangle, no triangle-soup
        // expansion and no LOD clustering (the tiled path spends ~1 s per 300k triangles).
        int          (*set_surface_indexed_blocks)(uint64_t, const double*, uint64_t, const uint32_t*,
                                                   uint64_t, const uint8_t*, const uint32_t*, uint32_t,
                                                   uint64_t, uint64_t) = nullptr;
        int          (*set_surface_indexed_blocks_vc)(uint64_t, const double*, uint64_t, const uint32_t*,
                                                   uint64_t, const uint8_t*, const uint32_t*, uint32_t,
                                                   uint64_t, uint64_t) = nullptr;
        int (*set_surface_complete)(uint64_t,const double*,uint64_t,const uint32_t*,uint64_t,
            const uint8_t*,const uint8_t*,const uint32_t*,uint32_t,const double*,const uint8_t*,uint64_t,
            uint64_t,uint64_t)=nullptr;
        // Crisp shaded-classification surface (VTK parity): positions + faces + a per-vertex
        // class id + the palettes + the mixed-face list; the engine builds the crisp facet
        // base and the barycentric mixed-class overlay itself.
        int (*set_shaded_class_surface)(uint64_t, const double*, uint64_t, const int32_t*, uint64_t,
                                        const uint8_t*, const uint8_t*, const int32_t*,
                                        uint64_t) = nullptr;
        int (*surface_elevation_color)(uint64_t, int, float, float, float) = nullptr;
        int (*set_crisp_shading_parameters)(uint64_t, float, float, float, float, float) = nullptr;
        // Compact form (shared vertices + one index buffer + coverage points): ~28 B per point
        // of VRAM, so a 50M-point full-detail cut fits a 4 GB GPU.
        int (*set_shaded_class_surface_indexed)(uint64_t, const double*, uint64_t, const int32_t*,
                                                uint64_t, const uint8_t*, const uint8_t*,
                                                const double*, const uint8_t*, uint64_t) = nullptr;
        bool warned_vertex_fallback = false;
        int (*set_shaded_class_surface_blocks)(uint64_t,const double*,uint64_t,const int32_t*,const uint32_t*,uint32_t)=nullptr;
        int (*surface_tiles_configure)(uint64_t, uint64_t, uint64_t, uint64_t) = nullptr;
        int (*surface_tiles_release)(uint64_t) = nullptr;
        int (*surface_tiles_active)(uint64_t) = nullptr;
        int (*surface_tiles_upload)(uint64_t, void*, uint32_t) = nullptr;
        int (*surface_tiles_evict)(uint64_t, const uint64_t*, uint32_t) = nullptr;
        int (*surface_tiles_set_palette)(uint64_t, const uint8_t*, float) = nullptr;
        int (*surface_tiles_set_draw_list)(uint64_t, const void*, uint32_t) = nullptr;
        int (*surface_tiles_get_stats)(uint64_t, uint64_t*, uint32_t) = nullptr;
        int (*surface_tiles_patch_classes)(uint64_t, uint64_t, const uint32_t*, const uint8_t*, uint32_t,
                                           const uint32_t*, const uint8_t*, uint32_t) = nullptr;
        int          (*set_surface_revision)(uint64_t, uint64_t) = nullptr;
        int          (*set_streaming_surface_active)(uint64_t, int) = nullptr;
        uint64_t     (*get_surface_total_triangle_count)(uint64_t) = nullptr;
        int          (*get_surface_state)(uint64_t) = nullptr;
        int          (*set_surface_lod_error_pixels)(uint64_t, float) = nullptr;
        uint32_t     (*get_surface_lod)(uint64_t, int*) = nullptr;
        uint64_t     (*get_point_capacity)(uint64_t) = nullptr;
        uint64_t     (*get_arena_uploaded_tiles)(uint64_t) = nullptr;
        int          (*set_point_draw_ranges)(uint64_t, const uint32_t*,
                                              const uint32_t*, uint32_t) = nullptr;
        int          (*clear_point_draw_ranges)(uint64_t) = nullptr;
        uint32_t     (*get_point_draw_range_count)(uint64_t) = nullptr;
        uint32_t     (*get_point_draw_range_points)(uint64_t) = nullptr;
        int          (*set_camera_lookat)(uint64_t, double, double, double,
                                          double, double, double,
                                          double, double, double) = nullptr;
        int          (*set_camera_view_up)(uint64_t, double, double, double) = nullptr;
        int          (*set_camera_ortho)(uint64_t, double, double, double, double, double,
                                         double, double, double, double, double) = nullptr;
        int          (*set_point_luts)(uint64_t, const uint8_t*, const uint8_t*,
                                       const uint8_t*) = nullptr;
        int          (*set_classification_lut)(uint64_t, const uint8_t*) = nullptr;
        int          (*set_clip_volume)(uint64_t, const float*, uint32_t) = nullptr;
        int          (*set_class_visibility)(uint64_t, const uint8_t*) = nullptr;
        int          (*set_render_origin)(uint64_t, double, double, double) = nullptr;
        int          (*resize)(uint64_t, uint32_t, uint32_t) = nullptr;
        int          (*render)(uint64_t) = nullptr;
        int          (*set_clear_color)(uint64_t, float, float, float, float) = nullptr;
        int          (*get_clear_color)(uint64_t, float*) = nullptr;
        int          (*capture_frame)(uint64_t, uint8_t*, uint64_t,
                                      uint32_t*, uint32_t*) = nullptr;
        int          (*get_last_mvp)(uint64_t, float*) = nullptr;
        uint64_t     (*get_point_count)(uint64_t) = nullptr;
        uint64_t     (*get_point_position_upload_count)(uint64_t) = nullptr;
        uint64_t     (*get_point_draw_call_count)(uint64_t) = nullptr;
        const char*  (*get_device_name)(uint64_t) = nullptr;
        uint64_t     (*get_vram_bytes)(uint64_t) = nullptr;
        uint64_t     (*get_device_vram_budget)(uint64_t) = nullptr;
        uint32_t     (*get_api_version)(uint64_t) = nullptr;
        double       (*get_gpu_frame_time_ms)(uint64_t) = nullptr;
        int          (*get_frame_stats)(uint64_t, uint64_t*, uint64_t*, uint64_t*) = nullptr;
        const char*  (*last_error)(void) = nullptr;
    };

    template <typename Fn>
    bool resolve(Fn& fn, const char* name, std::string& err);

    bool resolve_all(std::string& err);
    std::string engine_error() const;

    Api api_;
    uint64_t handle_ = 0;
    double render_origin_[3] = {0, 0, 0};   // last set_render_origin(), for clip planes
    std::string dll_path_;
    std::string error_;
    bool ready_ = false;
    // Cached capture extent. Learned from the engine, invalidated on resize.
    uint32_t capture_w_ = 0;
    uint32_t capture_h_ = 0;
#if defined(_WIN32)
    HWND hwnd_ = nullptr;
    bool owns_renderer_ = true;
    bool window_class_registered_ = false;
    uint32_t win_w_ = 0;
    uint32_t win_h_ = 0;
#endif
};

// ========================================================================
// implementations
// ========================================================================

// ------------------------------------------------------------------------
// EngineFrame
// ------------------------------------------------------------------------
inline const uint8_t* EngineFrame::at(uint32_t x, uint32_t y) const {
    if (x >= width || y >= height) return nullptr;
    const std::size_t i = (static_cast<std::size_t>(y) * width + x) * 4u;
    return (i + 3 < rgba.size()) ? &rgba[i] : nullptr;
}

inline uint64_t EngineFrame::count_not_equal(uint8_t r, uint8_t g, uint8_t b,
                                             int tol) const {
    uint64_t n = 0;
    for (std::size_t i = 0; i + 3 < rgba.size(); i += 4) {
        const int dr = static_cast<int>(rgba[i]) - static_cast<int>(r);
        const int dg = static_cast<int>(rgba[i + 1]) - static_cast<int>(g);
        const int db = static_cast<int>(rgba[i + 2]) - static_cast<int>(b);
        if (dr > tol || dr < -tol || dg > tol || dg < -tol ||
            db > tol || db < -tol)
            ++n;
    }
    return n;
}

inline uint64_t EngineFrame::count_where(bool (*pred)(uint8_t, uint8_t, uint8_t),
                                         int*) const {
    uint64_t n = 0;
    for (std::size_t i = 0; i + 3 < rgba.size(); i += 4)
        if (pred(rgba[i], rgba[i + 1], rgba[i + 2])) ++n;
    return n;
}

// ------------------------------------------------------------------------
// VulkanEngine - DLL discovery
// ------------------------------------------------------------------------
inline std::vector<std::string> VulkanEngine::dll_candidates() {
    std::vector<std::string> v;
    // 1. explicit override
    if (const char* env = std::getenv("LNT_VULKAN_DLL")) {
        if (env[0] != '\0') v.emplace_back(env);
    }
    // 2. the engine's own MSVC build tree, relative to several plausible roots
    static const char* tails[] = {
        "build_msvc/naksha_vulkan.dll",
        "native/naksha_vulkan/build_msvc/naksha_vulkan.dll",
        "../naksha_vulkan/build_msvc/naksha_vulkan.dll",
        "../../naksha_vulkan/build_msvc/naksha_vulkan.dll",
        "naksha_vulkan.dll",
    };
    for (const char* t : tails) v.emplace_back(t);
    return v;
}

inline std::string VulkanEngine::find_dll() {
#if defined(_WIN32)
    for (const std::string& c : dll_candidates()) {
        const DWORD a = GetFileAttributesA(c.c_str());
        if (a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY))
            return c;
    }
    return std::string();
#else
    return std::string();
#endif
}

// ------------------------------------------------------------------------
// VulkanEngine - loading
// ------------------------------------------------------------------------
template <typename Fn>
inline bool VulkanEngine::resolve(Fn& fn, const char* name, std::string& err) {
#if defined(_WIN32)
    fn = reinterpret_cast<Fn>(reinterpret_cast<void*>(
        GetProcAddress(api_.lib, name)));
    if (fn == nullptr) {
        // Name the export that is missing rather than a generic load failure:
        // an older DLL is the realistic cause, and the report should say which
        // capability is absent.
        err = std::string("export not found: ") + name;
        return false;
    }
    return true;
#else
    (void)fn; (void)name;
    err = "the Vulkan engine bridge is Windows-only (LoadLibrary/GetProcAddress)";
    return false;
#endif
}

inline bool VulkanEngine::resolve_all(std::string& err) {
    struct Binding { const char* name; };
    // Resolved as a REQUIRED core set. Anything the render path calls must be
    // present or the whole path is reported unavailable, so a missing export
    // can never turn into a silent no-op later.
#define LNT_RESOLVE(member, export_name)                                       \
    do { if (!resolve(api_.member, export_name, err)) return false; } while (0)

    LNT_RESOLVE(is_available, "nkv_is_available");
    LNT_RESOLVE(create_renderer, "nkv_create_renderer");
    LNT_RESOLVE(destroy_renderer, "nkv_destroy_renderer");
    LNT_RESOLVE(clear, "nkv_clear");
    LNT_RESOLVE(set_point_cloud, "nkv_set_point_cloud");
    LNT_RESOLVE(set_point_cloud_visible, "nkv_set_point_cloud_visible");
    LNT_RESOLVE(set_display_mode, "nkv_set_display_mode");
    LNT_RESOLVE(set_point_size_params, "nkv_set_point_size_params");
    LNT_RESOLVE(reserve_point_capacity, "nkv_reserve_point_capacity");
    LNT_RESOLVE(upload_point_tile, "nkv_upload_point_tile");
    LNT_RESOLVE(get_point_capacity, "nkv_get_point_capacity");
    LNT_RESOLVE(get_arena_uploaded_tiles, "nkv_get_arena_uploaded_tiles");
    LNT_RESOLVE(set_point_draw_ranges, "nkv_set_point_draw_ranges");
    LNT_RESOLVE(clear_point_draw_ranges, "nkv_clear_point_draw_ranges");
    LNT_RESOLVE(get_point_draw_range_count, "nkv_get_point_draw_range_count");
    LNT_RESOLVE(get_point_draw_range_points, "nkv_get_point_draw_range_points");
    LNT_RESOLVE(set_camera_lookat, "nkv_set_camera_lookat");
    LNT_RESOLVE(set_camera_view_up, "nkv_set_camera_view_up");
    LNT_RESOLVE(set_point_luts, "nkv_set_point_luts");
    // Optional for an older/headless engine DLL; the wrapper reports failure.
    { std::string optional_error;
      resolve(api_.set_classification_lut, "nkv_set_classification_lut", optional_error); }
    { std::string optional_error;
      resolve(api_.set_class_visibility, "nkv_set_class_visibility", optional_error); }
    { std::string optional_error;     // absent in an engine DLL built before shading modes
      resolve(api_.set_shading_mode, "nkv_set_shading_mode", optional_error); }
    { std::string optional_error;
      resolve(api_.upload_point_tile_ex, "nkv_upload_point_tile_ex", optional_error); }
    { std::string optional_error;     // the tiled-surface path (older than the shading API)
      resolve(api_.set_surface_tiled, "nkv_set_surface_tiled", optional_error);
      resolve(api_.set_surface_revision, "nkv_set_surface_revision", optional_error);
      { std::string ignore; resolve(api_.set_surface_indexed_blocks, "nkv_set_surface_indexed_blocks", ignore);
        resolve(api_.set_surface_indexed_blocks_vc, "nkv_set_surface_indexed_blocks_vc", ignore);
        resolve(api_.set_surface_complete,"nkv_set_surface_complete",ignore);
        // Crisp shaded-classification path (VTK parity). Optional: an engine DLL that
        // predates it leaves the .lnt view on the legacy CPU-baked facet colours.
        resolve(api_.set_shaded_class_surface, "nkv_set_shaded_class_surface", ignore);
        resolve(api_.surface_elevation_color, "nkv_set_surface_elevation_color", ignore);
        resolve(api_.set_crisp_shading_parameters, "nkv_set_crisp_shading_parameters", ignore);
        resolve(api_.set_shaded_class_surface_indexed, "nkv_set_shaded_class_surface_indexed", ignore); }
      resolve(api_.set_streaming_surface_active, "nkv_set_streaming_surface_active", optional_error);
      resolve(api_.get_surface_total_triangle_count, "nkv_get_surface_total_triangle_count", optional_error);
      resolve(api_.get_surface_state, "nkv_get_surface_state", optional_error);
      resolve(api_.set_surface_lod_error_pixels, "nkv_set_surface_lod_error_pixels", optional_error);
      resolve(api_.get_surface_lod, "nkv_get_surface_lod", optional_error); }
    { std::string optional_error;     // absent in an engine DLL built before cross sections
      resolve(api_.set_clip_volume, "nkv_set_clip_volume", optional_error); }
    LNT_RESOLVE(set_camera_ortho, "nkv_set_camera_ortho");
    {std::string ignore;resolve(api_.set_shaded_class_surface_blocks,"nkv_set_shaded_class_surface_blocks",ignore);}
    {   // viewport-LOD tile arena: absent in engine DLLs built before it
        std::string ignore;
        resolve(api_.surface_tiles_configure, "nkv_surface_tiles_configure", ignore);
        resolve(api_.surface_tiles_release, "nkv_surface_tiles_release", ignore);
        resolve(api_.surface_tiles_active, "nkv_surface_tiles_active", ignore);
        resolve(api_.surface_tiles_upload, "nkv_surface_tiles_upload", ignore);
        resolve(api_.surface_tiles_evict, "nkv_surface_tiles_evict", ignore);
        resolve(api_.surface_tiles_set_palette, "nkv_surface_tiles_set_palette", ignore);
        resolve(api_.surface_tiles_set_draw_list, "nkv_surface_tiles_set_draw_list", ignore);
        resolve(api_.surface_tiles_get_stats, "nkv_surface_tiles_get_stats", ignore);
        resolve(api_.surface_tiles_patch_classes, "nkv_surface_tiles_patch_classes", ignore);
    }
    LNT_RESOLVE(set_render_origin, "nkv_set_render_origin");
    LNT_RESOLVE(resize, "nkv_resize");
    LNT_RESOLVE(render, "nkv_render");
    LNT_RESOLVE(set_clear_color, "nkv_set_clear_color");
    LNT_RESOLVE(get_clear_color, "nkv_get_clear_color");
    LNT_RESOLVE(capture_frame, "nkv_capture_frame");
    LNT_RESOLVE(get_last_mvp, "nkv_get_last_mvp");        LNT_RESOLVE(get_point_count, "nkv_get_point_count");
    LNT_RESOLVE(get_point_position_upload_count, "nkv_get_point_position_upload_count");
    LNT_RESOLVE(get_point_draw_call_count, "nkv_get_point_draw_call_count");
    LNT_RESOLVE(get_device_name, "nkv_get_device_name");
    LNT_RESOLVE(get_vram_bytes, "nkv_get_vram_bytes");
    LNT_RESOLVE(get_device_vram_budget, "nkv_get_device_vram_budget");
    LNT_RESOLVE(get_api_version, "nkv_get_api_version");
    LNT_RESOLVE(get_gpu_frame_time_ms, "nkv_get_gpu_frame_time_ms");
    LNT_RESOLVE(get_frame_stats, "nkv_get_frame_stats");
    LNT_RESOLVE(last_error, "nkv_last_error");
#undef LNT_RESOLVE
    return true;
}

inline bool VulkanEngine::load(const std::string& dll_path, std::string& err) {
    error_.clear();
    dll_path_ = dll_path;
#if defined(_WIN32)
    if (dll_path.empty()) {
        err = "no naksha_vulkan.dll found (set LNT_VULKAN_DLL or build "
              "native/naksha_vulkan)";
        error_ = err;
        return false;
    }
    api_.lib = ::LoadLibraryExA(dll_path.c_str(), nullptr,
                               LOAD_WITH_ALTERED_SEARCH_PATH);
    if (api_.lib == nullptr) {
        err = "LoadLibrary failed for '" + dll_path +
              "' (GetLastError=" + std::to_string(GetLastError()) + ")";
        error_ = err;
        return false;
    }
    if (!resolve_all(err)) {
        error_ = err;
        return false;
    }
    api_.ready = true;
    ready_ = true;
    return true;
#else
    (void)dll_path;
    err = "the Vulkan engine bridge is Windows-only";
    error_ = err;
    return false;
#endif
}

inline std::string VulkanEngine::engine_error() const {
    if (api_.last_error) {
        const char* e = api_.last_error();
        if (e != nullptr && e[0] != '\0') return std::string(e);
    }
    return std::string("no engine error text");
}

inline bool VulkanEngine::is_available() const {
    if (!ready_ || api_.is_available == nullptr) return false;
    return api_.is_available() != 0;
}

// ------------------------------------------------------------------------
// VulkanEngine - device / window
// ------------------------------------------------------------------------
inline bool VulkanEngine::open(uint32_t width, uint32_t height, bool hidden,
                              bool validation, std::string& err, void* parent) {
    if (!ready_) {
        err = "engine not loaded";
        return false;
    }
    if (handle_ != 0) {
        err = "engine already open";
        return false;
    }
    if (!is_available()) {
        err = "no Vulkan loader or no physical device on this machine";
        error_ = err;
        return false;
    }
#if defined(_WIN32)
    // A real window is required because the engine's swapchain is bound to one.
    // HIDDEN is what makes this testable: nkv_capture_frame renders offscreen
    // into its own target, so pixel verification needs no presented frame and
    // no visible UI.
    HINSTANCE inst = GetModuleHandleA(nullptr);
    static const char* kClass = "LntVulkanEngineHostWindow";
    WNDCLASSA wc = {};
    wc.lpfnWndProc = DefWindowProcA;
    wc.hInstance = inst;
    wc.lpszClassName = kClass;
    if (RegisterClassA(&wc) != 0 || GetLastError() == ERROR_CLASS_ALREADY_EXISTS)
        window_class_registered_ = true;
    DWORD style = WS_OVERLAPPEDWINDOW;
    if (hidden) style &= ~WS_VISIBLE;
    HWND parent_hwnd = static_cast<HWND>(parent);
    if (parent_hwnd != nullptr) style = WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS;
    hwnd_ = CreateWindowExA(0, kClass, "lnt engine host", style,
                            parent_hwnd ? 0 : CW_USEDEFAULT, parent_hwnd ? 0 : CW_USEDEFAULT,
                            static_cast<int>(width), static_cast<int>(height),
                            parent_hwnd, nullptr, inst, nullptr);
    if (hwnd_ == nullptr) {
        err = "CreateWindowEx failed (GetLastError=" +
              std::to_string(GetLastError()) + ")";
        error_ = err;
        return false;
    }
    win_w_ = width;
    win_h_ = height;
    handle_ = api_.create_renderer(reinterpret_cast<void*>(hwnd_), width, height,
                                   validation ? 1 : 0);
#else
    (void)hidden; (void)validation;
    handle_ = 0;
#endif
    if (handle_ == 0) {
        err = "nkv_create_renderer failed: " + engine_error();
        error_ = err;
        return false;
    }
    // The render origin is the origin of the frame UBO and of every uploaded
    // position. Leaving it at (0,0,0) for a UTM-magnitude dataset wastes the
    // float32 mantissa, so the caller is expected to pin it to the dataset
    // centre (which the .lnt header supplies exactly).
    return true;
}

inline bool VulkanEngine::open_on_window(void* hwnd, uint32_t width, uint32_t height,
                                         std::string& err) {
    if (!ready_) { err = "engine not loaded"; return false; }
    if (handle_ != 0) { err = "engine already open"; return false; }
#if defined(_WIN32)
    handle_ = api_.create_renderer(hwnd, width, height, 0);
#else
    (void)hwnd;
#endif
    if (handle_ == 0) {
        err = "nkv_create_renderer failed: " + engine_error();
        error_ = err;
        return false;
    }
    owns_renderer_ = true;
    win_w_ = width;
    win_h_ = height;
    return true;
}

inline bool VulkanEngine::attach_existing(uint64_t handle) {
    if (!ready_ || handle == 0 || handle_ != 0) return false;
    handle_ = handle;
    owns_renderer_ = false;
    return true;
}

inline void VulkanEngine::close() {
    if (!owns_renderer_) {          // borrowed handle: detach only
        handle_ = 0;
        owns_renderer_ = true;
        return;
    }
    if (handle_ != 0 && api_.destroy_renderer != nullptr) {
        api_.destroy_renderer(handle_);   // blocks on GPU idle internally
        handle_ = 0;
    }
#if defined(_WIN32)
    if (hwnd_ != nullptr) {
        DestroyWindow(hwnd_);
        hwnd_ = nullptr;
    }
    if (window_class_registered_) {
        UnregisterClassA("LntVulkanEngineHostWindow", GetModuleHandleA(nullptr));
        window_class_registered_ = false;
    }
#endif
}

inline VulkanEngine::~VulkanEngine() {
    close();
#if defined(_WIN32)
    if (api_.lib != nullptr) {
        FreeLibrary(api_.lib);
        api_.lib = nullptr;
    }
#endif
    api_.ready = false;
    ready_ = false;
}

// ------------------------------------------------------------------------
// VulkanEngine - device introspection
// ------------------------------------------------------------------------
inline std::string VulkanEngine::device_name() const {
    if (handle_ == 0 || api_.get_device_name == nullptr) return std::string();
    const char* n = api_.get_device_name(handle_);
    return n != nullptr ? std::string(n) : std::string();
}
inline uint64_t VulkanEngine::vram_bytes() const {
    return (handle_ && api_.get_vram_bytes) ? api_.get_vram_bytes(handle_) : 0;
}
inline uint64_t VulkanEngine::vram_budget() const {
    return (handle_ && api_.get_device_vram_budget)
               ? api_.get_device_vram_budget(handle_) : 0;
}
inline uint32_t VulkanEngine::api_version() const {
    return (handle_ && api_.get_api_version) ? api_.get_api_version(handle_) : 0;
}

// ------------------------------------------------------------------------
// VulkanEngine - arena / draw ranges
// ------------------------------------------------------------------------
inline int VulkanEngine::reserve_point_capacity(uint64_t capacity_points) {
    if (handle_ == 0 || api_.reserve_point_capacity == nullptr) return 0;
    return api_.reserve_point_capacity(handle_, capacity_points);
}

inline bool VulkanEngine::upload_point_tile(uint64_t first, uint64_t count,
                                           const double* xyz_world_f64,
                                           const uint8_t* classification,
                                           const float* intensity,
                                           const uint8_t* rgb8) {
    if (handle_ == 0 || api_.upload_point_tile == nullptr) return false;
    if (rgb8 != nullptr && api_.upload_point_tile_ex != nullptr)
        return api_.upload_point_tile_ex(handle_, first, count, xyz_world_f64, classification,
                                         intensity, nullptr, rgb8) != 0;
    return api_.upload_point_tile(handle_, first, count, xyz_world_f64,
                                  classification, intensity, nullptr) != 0;
}
inline bool VulkanEngine::upload_surface(const SurfaceMesh& mesh, uint64_t revision) {
    if (handle_ == 0 || !has_surface() || (mesh.faces.empty()&&mesh.coverage_positions.empty())) return false;
    if (api_.set_surface_revision) api_.set_surface_revision(handle_, revision);
    if(api_.set_surface_complete && !mesh.coverage_positions.empty())
        return api_.set_surface_complete(handle_,mesh.positions.data(),mesh.vertices(),
            reinterpret_cast<const uint32_t*>(mesh.faces.data()),mesh.triangles(),mesh.colors.data(),
            mesh.vertex_colors.empty()?nullptr:mesh.vertex_colors.data(),mesh.blocks.data(),uint32_t(mesh.blocks.size()),
            mesh.coverage_positions.data(),mesh.coverage_colors.data(),mesh.coverage_positions.size()/3,0,revision)!=0;
    if(!mesh.coverage_positions.empty()) {
        if(!api_.warned_vertex_fallback) {
            std::fprintf(stderr,"[LNT surface] old engine cannot retain unmeshed points; full-detail upload refused\n");
            api_.warned_vertex_fallback=true;
        }
        return false;
    }
    if (!mesh.vertex_colors.empty() && !mesh.blocks.empty()) {
        if (api_.set_surface_indexed_blocks_vc)
            return api_.set_surface_indexed_blocks_vc(handle_, mesh.positions.data(), mesh.vertices(),
                reinterpret_cast<const uint32_t*>(mesh.faces.data()), mesh.triangles(),
                mesh.vertex_colors.data(), mesh.blocks.data(), uint32_t(mesh.blocks.size()), 0, revision) != 0;
        if (!api_.warned_vertex_fallback) {
            std::fprintf(stderr, "[LNT surface] engine lacks vertex-colour export; using facet fallback\n");
            api_.warned_vertex_fallback = true;
        }
    }
    if (api_.set_surface_indexed_blocks && !mesh.blocks.empty())
        return api_.set_surface_indexed_blocks(handle_, mesh.positions.data(), mesh.vertices(),
                                               reinterpret_cast<const uint32_t*>(mesh.faces.data()),
                                               mesh.triangles(), mesh.colors.data(),
                                               mesh.blocks.data(), uint32_t(mesh.blocks.size()),
                                               /*dataset_revision=*/0, revision) != 0;
    return api_.set_surface_tiled(handle_, mesh.positions.data(), mesh.vertices(), mesh.faces.data(),
                                  mesh.triangles(), mesh.colors.data(), /*tile_count=*/0,
                                  /*base_cell_m=*/0.0f, /*is_preview=*/0, /*dataset_revision=*/0,
                                  revision) != 0;
}
inline bool VulkanEngine::upload_shaded_class_surface(const SurfaceMesh& mesh,
                                                     const uint8_t* class_color_lut,
                                                     uint64_t revision) {
    if (handle_ == 0 || !has_shaded_class_surface()) return false;
    if (mesh.faces.empty() || mesh.positions.empty() || mesh.vertex_class.empty() ||
        class_color_lut == nullptr) return false;
    // vertex_class has one byte per vertex; reject a mismatched stream rather than read past it.
    if (mesh.vertex_class.size() != mesh.vertices()) return false;
    if (api_.set_surface_revision) api_.set_surface_revision(handle_, revision);
    if (api_.set_shaded_class_surface_indexed) {
        const uint64_t coverage = uint64_t(mesh.coverage_positions.size() / 3);
        if (api_.set_shaded_class_surface_indexed(
                handle_, mesh.positions.data(), mesh.vertices(), mesh.faces.data(),
                mesh.triangles(), mesh.vertex_class.data(), class_color_lut,
                coverage ? mesh.coverage_positions.data() : nullptr,
                coverage ? mesh.coverage_colors.data() : nullptr, coverage) != 0) {
            if(api_.set_shaded_class_surface_blocks&&!mesh.blocks.empty())
                api_.set_shaded_class_surface_blocks(handle_,mesh.positions.data(),mesh.vertices(),
                    mesh.faces.data(),mesh.blocks.data(),uint32_t(mesh.blocks.size()));
            return true;
        }
        std::fprintf(stderr, "[LNT surface] compact crisp upload failed: %s\n", last_error());
        return false;
    }
    if (api_.set_shaded_class_surface == nullptr) return false;
    return api_.set_shaded_class_surface(
               handle_, mesh.positions.data(), mesh.vertices(), mesh.faces.data(),
               mesh.triangles(), mesh.vertex_class.data(), class_color_lut,
               mesh.mixed_faces.empty() ? nullptr : mesh.mixed_faces.data(),
               uint64_t(mesh.mixed_faces.size())) != 0;
}
inline bool VulkanEngine::set_crisp_shading_parameters(double azimuth_deg, double sharpness_raw,
                                                       double ambient, double key_intensity,
                                                       double fill_intensity) {
    if (handle_ == 0 || api_.set_crisp_shading_parameters == nullptr) return false;
    return api_.set_crisp_shading_parameters(
               handle_, float(azimuth_deg), float(sharpness_raw), float(ambient),
               float(key_intensity), float(fill_intensity)) != 0;
}
inline bool VulkanEngine::show_surface(bool on) {
    if (handle_ == 0 || api_.set_streaming_surface_active == nullptr) return false;
    return api_.set_streaming_surface_active(handle_, on ? 1 : 0) != 0;
}
inline bool VulkanEngine::set_surface_lod_error_pixels(float pixels) {
    return handle_ && api_.set_surface_lod_error_pixels &&
           api_.set_surface_lod_error_pixels(handle_, pixels) != 0;
}
inline uint32_t VulkanEngine::surface_lod() const {
    return handle_ && api_.get_surface_lod ? api_.get_surface_lod(handle_, nullptr) : 0;
}
inline uint64_t VulkanEngine::surface_triangles() const {
    return (handle_ && api_.get_surface_total_triangle_count)
               ? api_.get_surface_total_triangle_count(handle_) : 0;
}
inline int VulkanEngine::surface_state() const {
    return (handle_ && api_.get_surface_state) ? api_.get_surface_state(handle_) : 0;
}
inline bool VulkanEngine::set_shading_mode(const ShadingUniforms& config) {
    if (handle_ == 0 || api_.set_shading_mode == nullptr) return false;
    return api_.set_shading_mode(handle_, &config) != 0;
}

inline uint64_t VulkanEngine::point_capacity() const {
    return (handle_ && api_.get_point_capacity) ? api_.get_point_capacity(handle_) : 0;
}
inline uint64_t VulkanEngine::arena_uploaded_tiles() const {
    return (handle_ && api_.get_arena_uploaded_tiles)
               ? api_.get_arena_uploaded_tiles(handle_) : 0;
}

inline bool VulkanEngine::set_point_draw_ranges(const std::vector<uint32_t>& firsts,
                                               const std::vector<uint32_t>& counts) {
    if (handle_ == 0 || api_.set_point_draw_ranges == nullptr) return false;
    if (firsts.size() != counts.size()) return false;
    if (firsts.empty()) return clear_point_draw_ranges();
    return api_.set_point_draw_ranges(handle_, firsts.data(), counts.data(),
                                      static_cast<uint32_t>(firsts.size())) != 0;
}

inline bool VulkanEngine::clear_point_draw_ranges() {
    if (handle_ == 0 || api_.clear_point_draw_ranges == nullptr) return false;
    return api_.clear_point_draw_ranges(handle_) != 0;
}
inline uint32_t VulkanEngine::draw_range_count() const {
    return (handle_ && api_.get_point_draw_range_count)
               ? api_.get_point_draw_range_count(handle_) : 0u;
}
inline uint32_t VulkanEngine::draw_range_points() const {
    return (handle_ && api_.get_point_draw_range_points)
               ? api_.get_point_draw_range_points(handle_) : 0u;
}

// ------------------------------------------------------------------------
// VulkanEngine - point cloud / display
// ------------------------------------------------------------------------
inline bool VulkanEngine::set_point_cloud(const double* xyz, const uint8_t* rgb,
                                         const uint8_t* classification,
                                         const float* intensity, uint64_t count) {
    if (handle_ == 0 || api_.set_point_cloud == nullptr) return false;
    return api_.set_point_cloud(handle_, xyz, rgb, classification, intensity,
                                count) != 0;
}
inline bool VulkanEngine::set_point_cloud_visible(bool visible) {
    if (handle_ == 0 || api_.set_point_cloud_visible == nullptr) return false;
    return api_.set_point_cloud_visible(handle_, visible ? 1 : 0) != 0;
}
inline bool VulkanEngine::set_display_mode(int mode) {
    if (handle_ == 0 || api_.set_display_mode == nullptr) return false;
    return api_.set_display_mode(handle_, mode) != 0;
}
inline bool VulkanEngine::set_point_size_params(float footprint_m, float min_px,
                                               float max_px,
                                               float class_intensity_mix) {
    if (handle_ == 0 || api_.set_point_size_params == nullptr) return false;
    return api_.set_point_size_params(handle_, footprint_m, min_px, max_px,
                                      class_intensity_mix) != 0;
}

// ------------------------------------------------------------------------
// VulkanEngine - camera / frame
// ------------------------------------------------------------------------
inline bool VulkanEngine::set_point_luts(const uint8_t* c, const uint8_t* e, const uint8_t* i) {
    if (handle_ == 0 || api_.set_point_luts == nullptr) return false;
    return api_.set_point_luts(handle_, c, e, i) != 0;
}
inline bool VulkanEngine::set_clip_volume(const float* planes, uint32_t count) {
    if (handle_ == 0 || api_.set_clip_volume == nullptr) return false;
    if (count > 4) count = 4;
    return api_.set_clip_volume(handle_, planes, count) != 0;
}
inline bool VulkanEngine::set_classification_lut(const PackedClassColor lut[256]) {
    if (!handle_ || !lut) return false;
    if (api_.set_classification_lut)
        return api_.set_classification_lut(handle_, reinterpret_cast<const uint8_t*>(lut)) != 0;
    if (!api_.set_point_luts || !api_.set_class_visibility) return false;
    uint8_t rgb[256 * 3], visible[256];
    for (unsigned i = 0; i < 256; ++i) {
        rgb[i * 3] = lut[i].r;
        rgb[i * 3 + 1] = lut[i].g;
        rgb[i * 3 + 2] = lut[i].b;
        visible[i] = lut[i].visible ? 1 : 0;
    }
    return api_.set_point_luts(handle_, rgb, nullptr, nullptr) != 0 &&
           api_.set_class_visibility(handle_, visible) != 0;
}
inline bool VulkanEngine::set_camera_ortho(double cx, double cy, double cz, double dx,
                                           double dy, double dz, double parallel_scale,
                                           double aspect, double near_clip, double far_clip) {
    if (handle_ == 0 || api_.set_camera_ortho == nullptr) return false;
    return api_.set_camera_ortho(handle_, cx, cy, cz, dx, dy, dz, parallel_scale, aspect,
                                 near_clip, far_clip) != 0;
}
inline bool VulkanEngine::set_camera_lookat(double ex, double ey, double ez,
                                            double tx, double ty, double tz,
                                            double fov_y_degrees, double near_clip,
                                            double far_clip) {
    if (handle_ == 0 || api_.set_camera_lookat == nullptr) return false;
    return api_.set_camera_lookat(handle_, ex, ey, ez, tx, ty, tz, fov_y_degrees,
                                  near_clip, far_clip) != 0;
}
inline bool VulkanEngine::set_camera_view_up(double x, double y, double z) {
    if (handle_ == 0 || api_.set_camera_view_up == nullptr) return false;
    return api_.set_camera_view_up(handle_, x, y, z) != 0;
}
inline bool VulkanEngine::set_render_origin(double x, double y, double z) {
    if (handle_ == 0 || api_.set_render_origin == nullptr) return false;
    render_origin_[0] = x; render_origin_[1] = y; render_origin_[2] = z;
    return api_.set_render_origin(handle_, x, y, z) != 0;
}
inline bool VulkanEngine::set_clip_planes_world(const double* abcd, uint32_t count) {
    if (count == 0 || abcd == nullptr) return set_clip_volume(nullptr, 0);
    if (count > 4) count = 4;
    float f[16] = {};
    for (uint32_t i = 0; i < count; ++i) {
        const double* p = abcd + 4 * i;
        f[4 * i + 0] = float(p[0]);
        f[4 * i + 1] = float(p[1]);
        f[4 * i + 2] = float(p[2]);
        f[4 * i + 3] = float(p[3] + p[0] * render_origin_[0] + p[1] * render_origin_[1] +
                             p[2] * render_origin_[2]);
    }
    return set_clip_volume(f, count);
}
inline bool VulkanEngine::resize(uint32_t width, uint32_t height) {
    if (handle_ == 0 || api_.resize == nullptr) return false;
    // The capture extent follows the swapchain, so a resize invalidates it.
    capture_w_ = 0;
    capture_h_ = 0;
    return api_.resize(handle_, width, height) != 0;
}
inline int VulkanEngine::render() {
    if (handle_ == 0 || api_.render == nullptr) return 0;
    return api_.render(handle_);
}
inline bool VulkanEngine::set_clear_color(float r, float g, float b, float a) {
    if (handle_ == 0 || api_.set_clear_color == nullptr) return false;
    return api_.set_clear_color(handle_, r, g, b, a) != 0;
}
inline bool VulkanEngine::get_clear_color(float out_rgba[4]) const {
    if (handle_ == 0 || api_.get_clear_color == nullptr) return false;
    return api_.get_clear_color(handle_, out_rgba) != 0;
}

inline bool VulkanEngine::capture_frame(EngineFrame& out) {
    if (handle_ == 0 || api_.capture_frame == nullptr) return false;
    // The engine reports the TRUE extents even when the capacity is short
    // (documented), which is exactly how a caller learns what to allocate. A
    // first probe call with a tiny buffer must therefore KEEP the w/h it
    // reports.
    //
    // This was a real bug here: the probe's failure was treated as "extent
    // unknown" and the WINDOW size substituted instead. The engine then filled
    // only its own (swapchain) extent and the rest of the buffer stayed
    // zero-initialised - 32016 phantom black pixels on a 640x480 window with a
    // 624x441 swapchain, which every pixel count then reported as "drawn" even
    // with an empty draw list. Measured, not reasoned: the colour histogram
    // showed exactly 275184 red + 32016 black = 640*480.
    uint32_t w = capture_w_, h = capture_h_;
    if (w == 0 || h == 0) {
        uint8_t probe[4] = {0, 0, 0, 0};
        api_.capture_frame(handle_, probe, sizeof(probe), &w, &h);   // rc ignored on purpose
        if (w == 0 || h == 0) return false;
        capture_w_ = w;
        capture_h_ = h;
    }
    out.width = w;
    out.height = h;
    out.rgba.assign(static_cast<std::size_t>(w) * h * 4u, 0);
    uint32_t w2 = 0, h2 = 0;
    if (api_.capture_frame(handle_, out.rgba.data(),
                           static_cast<uint64_t>(out.rgba.size()), &w2, &h2) == 0)
        return false;
    // A second call reporting a different extent means the swapchain changed
    // between the two calls, so the buffer no longer matches what was filled.
    if (w2 != w || h2 != h) {
        capture_w_ = 0;
        capture_h_ = 0;
        return false;
    }
    return true;
}

inline bool VulkanEngine::get_last_mvp(float out_mvp16[16]) {
    if (handle_ == 0 || api_.get_last_mvp == nullptr) return false;
    return api_.get_last_mvp(handle_, out_mvp16) != 0;
}

// ------------------------------------------------------------------------
// VulkanEngine - counters
// ------------------------------------------------------------------------
inline uint64_t VulkanEngine::point_count() const {
    return (handle_ && api_.get_point_count) ? api_.get_point_count(handle_) : 0;
}
inline uint64_t VulkanEngine::position_upload_count() const {
    return (handle_ && api_.get_point_position_upload_count)
               ? api_.get_point_position_upload_count(handle_) : 0;
}
// vkCmdDraw calls the point path actually issued. This is the authoritative
// answer to "did the draw list draw nothing?" - an empty arena draw list must
// produce ZERO calls, which a pixel count alone cannot distinguish from a
// camera aimed at empty space.
inline uint64_t VulkanEngine::point_draw_call_count() const {
    return (handle_ && api_.get_point_draw_call_count)
               ? api_.get_point_draw_call_count(handle_) : 0;
}
inline double VulkanEngine::gpu_frame_time_ms() const {
    return (handle_ && api_.get_gpu_frame_time_ms)
               ? api_.get_gpu_frame_time_ms(handle_) : -1.0;
}
inline bool VulkanEngine::frame_stats(uint64_t* rendered, uint64_t* skipped,
                                     uint64_t* recreates) const {
    if (handle_ == 0 || api_.get_frame_stats == nullptr) return false;
    return api_.get_frame_stats(handle_, rendered, skipped, recreates) != 0;
}
inline const char* VulkanEngine::last_error() const {
    if (api_.last_error == nullptr) return "";
    const char* e = api_.last_error();
    return e != nullptr ? e : "";
}

}  // namespace lnt

#endif  // LNT_VULKAN_ENGINE_H
