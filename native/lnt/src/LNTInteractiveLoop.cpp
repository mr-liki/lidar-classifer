// LNTInteractiveLoop.cpp - see include/LNTInteractiveLoop.h.

#include "../include/LNTInteractiveLoop.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace lnt {

namespace {
bool trace_on() { static const bool t = std::getenv("LNT_TRACE") != nullptr; return t; }
#define LNT_STAGE(s) do { if (trace_on()) { std::fprintf(stderr, "[LNT-TRACE] %s\n", s); std::fflush(stderr); } } while (0)
using Clock = std::chrono::steady_clock;
float ms_since(Clock::time_point t0) {
    return float(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
}
}  // namespace

void LntInteractiveLoop::attach(LntCameraController* controller, LntCacheManager* cache,
                                const double render_origin[3]) {
    controller_ = controller;
    cache_ = cache;
    for (int i = 0; i < 3; ++i) origin_[i] = render_origin[i];
}

void LntInteractiveLoop::set_viewport(uint32_t width, uint32_t height) {
    width_ = width ? width : 1;
    height_ = height ? height : 1;
}

void LntInteractiveLoop::fill_cache_stats() {
    if (!cache_) return;
    const LntCacheStats cs = cache_->stats();
    stats_.cache_bytes_used = std::size_t(cs.resident_bytes);
    stats_.cache_capacity_bytes = std::size_t(cs.budget_bytes);
    stats_.cache_peak_bytes = std::size_t(cs.peak_bytes);
    stats_.total_evictions = uint32_t(cs.evictions);
    stats_.pending_async_loads = uint32_t(cache_->pending());
}

bool LntInteractiveLoop::prepare(LntStreamer& streamer, float dt) {
    err_.clear();
    if (!controller_) { err_ = "loop has no camera controller"; return false; }
    controller_->update(dt);
    if (!controller_->apply_to(cam_, origin_, double(width_) / double(height_), height_)) {
        err_ = "degenerate camera basis";
        return false;
    }
    LntLodParams p = streamer.params();
    p.sse_threshold = controller_->sse_threshold();
    streamer.set_params(p);
    return true;
}

bool LntInteractiveLoop::update_frame(LntStreamer& streamer, float dt) {
    const auto t0 = Clock::now();
    if (!prepare(streamer, dt)) return false;
    const auto ts = Clock::now();
    const std::size_t cut = streamer.update_visible_nodes(cam_);
    stats_.streamer_update_ms = ms_since(ts);
    stats_.active_nodes = uint32_t(cut);
    stats_.wanted_points = uint32_t(streamer.stats().wanted_points);
    // No engine: "rendered" is what the resident part of the cut would draw.
    stats_.total_points_rendered = stats_.wanted_points;
    stats_.sse_threshold = controller_->sse_threshold();
    fill_cache_stats();
    stats_.frame_time_ms = ms_since(t0);
    return true;
}

bool LntInteractiveLoop::render_frame(VulkanEngine& engine, LntScene& scene,
                                      LntStreamer& streamer, float dt, bool update_points) {
    presented_=false;
    const auto t0 = Clock::now();
    if (!prepare(streamer, dt)) return false;

    LNT_STAGE("render_frame: set_camera_lookat");
    // The engine renders the same eye/target/lens the frustum was built from.
    // Eye/target alone do not define camera roll. Push the controller's view-up
    // first so rolled presets such as horizontal Side reach the Vulkan camera.
    if (!engine.set_camera_view_up(cam_.up[0], cam_.up[1], cam_.up[2])) {
        err_ = std::string("set_camera_view_up failed: ") + engine.last_error();
        return false;
    }
    if (cam_.ortho) {
        double d[3] = {cam_.target[0] - cam_.eye[0], cam_.target[1] - cam_.eye[1],
                       cam_.target[2] - cam_.eye[2]};
        const double n = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        if (n > 1e-12) { d[0] /= n; d[1] /= n; d[2] /= n; }
        if (!engine.set_camera_ortho(cam_.target[0], cam_.target[1], cam_.target[2], d[0], d[1],
                                     d[2], cam_.ortho_half_height, cam_.aspect, cam_.near_clip,
                                     cam_.far_clip)) {
            err_ = std::string("set_camera_ortho failed: ") + engine.last_error();
            return false;
        }
    } else if (!engine.set_camera_lookat(cam_.eye[0], cam_.eye[1], cam_.eye[2], cam_.target[0],
                                         cam_.target[1], cam_.target[2], cam_.fov_y_deg,
                                         cam_.near_clip, cam_.far_clip)) {
        err_ = std::string("set_camera_lookat failed: ") + engine.last_error();
        return false;
    }

    LNT_STAGE("render_frame: streamer.update_visible_nodes");
    const auto ts = Clock::now();
    if(update_points) {
    streamer.set_upload_budget_ms(upload_budget_ms_);
    std::string uerr;
    streamer.update_visible_nodes(engine, scene, cam_, uerr);
    stats_.streamer_update_ms = ms_since(ts);
    if (!uerr.empty()) { err_ = uerr; return false; }

    const LntStreamStats& ss = streamer.stats();
    stats_.active_nodes = uint32_t(ss.cut_nodes);
    stats_.wanted_points = uint32_t(ss.wanted_points);
    stats_.total_points_rendered = uint32_t(ss.cut_points);   // pushed (resident) only
    stats_.sse_threshold = controller_->sse_threshold();
    } else {
        // The point pass is hidden in Shading. Camera and surface presentation
        // still update every frame; point selection/uploads resume on mode exit.
        stats_.streamer_update_ms=0;
    }

    LNT_STAGE("render_frame: present");
    switch (present_) {
    case LntPresent::Render: {
        const int result=engine.render();
        if (result == 0) {
            err_ = std::string("render failed: ") + engine.last_error();
            return false;
        }
        presented_=result==2;
        break;
    }
    case LntPresent::Capture: {
        EngineFrame f;
        if (!engine.capture_frame(f)) {
            err_ = std::string("capture_frame failed: ") + engine.last_error();
            return false;
        }
        presented_=true;
        break;
    }
    case LntPresent::None: break;
    }
    fill_cache_stats();
    stats_.frame_time_ms = ms_since(t0);
    return true;
}

}  // namespace lnt
