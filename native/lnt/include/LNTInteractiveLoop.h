// LNTInteractiveLoop.h - the per-frame bridge: camera input -> streamer -> engine.
//
// One call per frame:
//   1. controller.update(dt), then build the LNTCamera (view, projection, frustum
//      planes) from it;
//   2. copy the controller's SSE threshold into the streamer;
//   3. LntStreamer::update_visible_nodes(): select the cut, pin + request missing
//      payloads (never blocking), upload newly resident tiles, and push the draw
//      ranges with nkv_set_point_draw_ranges (via LntScene::push_tiles);
//   4. present: engine.render() (or capture_frame() when asked, the blocking
//      readback path).
// update_frame() runs steps 1-3 without an engine, which is how the loop is
// exercised on a machine with no Vulkan device.
//
// The names are the project's real ones (VulkanEngine, LntStreamer, LntScene),
// not the LNTVulkanEngine / LNT::Streamer of the task text.

#ifndef LNT_INTERACTIVE_LOOP_H
#define LNT_INTERACTIVE_LOOP_H

#include <cstddef>
#include <array>
#include <vector>
#include <cstdint>
#include <string>

#include "LNTCacheManager.h"
#include "LNTCameraController.h"
#include "LNTStreamer.h"
#include "LNTVulkanEngine.h"

namespace lnt {

struct LNTStreamerStats {
    uint32_t active_nodes = 0;           // nodes in this frame's cut
    uint32_t total_points_rendered = 0;  // points in the ranges actually pushed
    float    frame_time_ms = 0.f;        // whole render_frame / update_frame call
    float    streamer_update_ms = 0.f;   // the streamer call (select, upload, push)
    std::size_t cache_bytes_used = 0;
    std::size_t cache_capacity_bytes = 0;
    uint32_t pending_async_loads = 0;
    uint32_t total_evictions = 0;
    // Extra, useful alongside the above.
    uint32_t wanted_points = 0;          // what the cut asks for once all is resident
    double   sse_threshold = 0.0;
    std::size_t cache_peak_bytes = 0;
};

enum class LntPresent { Render, Capture, None };

class LntInteractiveLoop {
public:
    // The loop does not own any of these.
    void attach(LntCameraController* controller, LntCacheManager* cache,
                const double render_origin[3]);
    void set_viewport(uint32_t width, uint32_t height);
    void set_present_mode(LntPresent m) { present_ = m; }
    // Time slice for GPU uploads per frame (ms); 0 = unlimited. Keeps a frame's
    // cost bounded when a camera move makes many tiles newly visible at once.
    void set_upload_budget_ms(double ms) { upload_budget_ms_ = ms; }
    void set_depth_samples(const std::vector<std::array<double, 3>>& points) { depth_samples_ = points; }
    void set_depth_enabled(bool enabled) { depth_enabled_ = enabled; }

    LntCameraController& controller() { return *controller_; }
    const LNTCamera& camera() const { return cam_; }
    const LNTStreamerStats& stats() const { return stats_; }
    bool presented() const { return presented_; }

    // Full frame with the engine. False on a camera / upload / presentation
    // failure (reason in error()).
    bool render_frame(VulkanEngine& engine, LntScene& scene, LntStreamer& streamer,
                      float dt, bool update_points = true);
    // Steps 1-3 only: camera, selection, cache requests. No engine.
    bool update_frame(LntStreamer& streamer, float dt);

    const std::string& error() const { return err_; }

private:
    bool prepare(LntStreamer& streamer, float dt);
    void fill_cache_stats();

    LntCameraController* controller_ = nullptr;
    LntCacheManager* cache_ = nullptr;
    double origin_[3] = {0.0, 0.0, 0.0};
    uint32_t width_ = 960, height_ = 720;
    LntPresent present_ = LntPresent::Render;
    bool presented_=false;
    double upload_budget_ms_ = 1.0;
    LNTCamera cam_;
    LNTStreamerStats stats_;
    std::string err_;
    bool depth_enabled_ = false;
    std::vector<std::array<double, 3>> depth_samples_;
    std::vector<double> depth_values_;
};

}  // namespace lnt

#endif  // LNT_INTERACTIVE_LOOP_H
