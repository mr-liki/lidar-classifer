// LNTCameraController.cpp - see include/LNTCameraController.h.

#include "../include/LNTCameraController.h"

#include <algorithm>
#include <cmath>

namespace lnt {

namespace {
constexpr double kPitchLimit = 89.0 * 3.14159265358979323846 / 180.0;
constexpr double kZoomRate = 8.0;     // 1/s: the glide closes ~63% of the gap in 125 ms
double clampd(double v, double lo, double hi) { return std::min(std::max(v, lo), hi); }
}  // namespace

LntCameraController::LntCameraController() = default;

void LntCameraController::set_target(double x, double y, double z) {
    target_[0] = x; target_[1] = y; target_[2] = z;
}

void LntCameraController::set_distance(double d) {
    dist_ = goal_dist_ = clampd(d, kMinDistance, kMaxDistance);
}

void LntCameraController::set_angles(double yaw, double pitch) {
    yaw_ = yaw;
    pitch_ = clampd(pitch, -kPitchLimit, kPitchLimit);
}

void LntCameraController::set_lens(double fov_y_deg, double near_clip, double far_clip) {
    fov_ = fov_y_deg; near_ = near_clip; far_ = far_clip;
}

void LntCameraController::set_ortho2d(bool on) {
    ortho2d_ = on;
    if (on) { yaw_ = 0.0; pitch_ = 0.0; }
}

void LntCameraController::orbit(float delta_yaw, float delta_pitch) {
    if (ortho2d_) return;               // locked to the plan view
    const double two_pi = 2.0 * 3.14159265358979323846;
    yaw_ = std::fmod(yaw_ + double(delta_yaw), two_pi);
    pitch_ = clampd(pitch_ + double(delta_pitch), -kPitchLimit, kPitchLimit);
}

void LntCameraController::eye(double out[3]) const {
    const double cp = std::cos(pitch_);
    out[0] = target_[0] + dist_ * cp * std::sin(yaw_);
    out[1] = target_[1] + dist_ * std::sin(pitch_);
    out[2] = target_[2] + dist_ * cp * std::cos(yaw_);
}

void LntCameraController::pan(float delta_x, float delta_y) {
    // View basis from the angles directly (no matrix needed): forward points
    // eye -> target, right = forward x up, camera-up = right x forward.
    const double cp = std::cos(pitch_), sp = std::sin(pitch_);
    const double fx = -cp * std::sin(yaw_), fy = -sp, fz = -cp * std::cos(yaw_);
    double rx = fy * 0.0 - fz * 1.0, ry = fz * 0.0 - fx * 0.0, rz = fx * 1.0 - fy * 0.0;
    const double rl = std::sqrt(rx * rx + ry * ry + rz * rz);   // > 0: pitch < 90 deg
    rx /= rl; ry /= rl; rz /= rl;
    const double ux = ry * fz - rz * fy, uy = rz * fx - rx * fz, uz = rx * fy - ry * fx;
    target_[0] += rx * delta_x + ux * delta_y;
    target_[1] += ry * delta_x + uy * delta_y;
    target_[2] += rz * delta_x + uz * delta_y;
}

void LntCameraController::zoom(float delta_distance) {
    goal_dist_ = clampd(goal_dist_ + double(delta_distance), kMinDistance, kMaxDistance);
}

bool LntCameraController::update(float dt) {
    if (dt <= 0.0f) return dist_ != goal_dist_;
    const double a = 1.0 - std::exp(-kZoomRate * double(dt));
    dist_ += (goal_dist_ - dist_) * a;
    if (std::fabs(goal_dist_ - dist_) < 1e-4) dist_ = goal_dist_;
    dist_ = clampd(dist_, kMinDistance, kMaxDistance);
    return dist_ != goal_dist_;
}

void LntCameraController::set_sse_threshold(double px) {
    sse_ = std::isfinite(px) ? clampd(px, kMinSse, kMaxSse) : sse_;
}

bool LntCameraController::apply_to(LNTCamera& cam, const double origin[3], double aspect,
                                   uint32_t screen_height_px) const {
    for (int i = 0; i < 3; ++i) { cam.target[i] = target_[i]; cam.origin[i] = origin[i]; }
    cam.up[0] = 0.0; cam.up[1] = 1.0; cam.up[2] = 0.0;
    cam.fov_y_deg = fov_;
    cam.near_clip = near_;
    cam.far_clip = far_;
    cam.ortho = ortho2d_;
    if (ortho2d_) {
        // Same visible extent at the target as the perspective camera would show at
        // `dist_`, as a parallel scale (half the visible height). The engine puts the
        // eye at the middle of [near, far] along the view direction, so the matrices
        // here are built from exactly that eye - identical volume, identical culling.
        const double tn = std::tan(fov_ * 3.14159265358979323846 / 360.0);
        cam.ortho_half_height = std::max(dist_ * tn, 1e-6);
        const double standoff = 0.5 * (near_ + far_);
        cam.eye[0] = target_[0];
        cam.eye[1] = target_[1];
        cam.eye[2] = target_[2] + standoff;
    } else {
        eye(cam.eye);
    }
    cam.aspect = aspect;
    cam.screen_height_px = screen_height_px;
    return cam.rebuild();
}

}  // namespace lnt
