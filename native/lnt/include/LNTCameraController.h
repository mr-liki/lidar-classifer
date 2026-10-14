// LNTCameraController.h - interactive orbit / pan / zoom camera + SSE dial.
//
// MATH. Spherical coordinates about a target anchor: yaw (about +Y), pitch
// (elevation, clamped to +-89 degrees so the view never goes parallel to the up
// vector and the basis stays unique) and distance. Y-up is deliberate: it is the
// up the engine and LNTCamera use (see LNTCamera::up). The eye is
//
//     eye = target + d * ( cos(p) * sin(y),  sin(p),  cos(p) * cos(y) )
//
// ORBIT moves the angles; PAN moves target AND eye together along the view's
// right and up axes (so the orientation and distance are unchanged); ZOOM sets a
// goal distance, clamped to [1, 10000] at the moment it is requested, and
// update(dt) approaches that goal exponentially - a smooth glide, with the
// clamp applied to the goal so the glide can never overshoot the range.
//
// SSE TUNING. The screen-space-error threshold is the LOD dial: 0 is full
// resolution (descend to the leaves), larger is coarser. It is clamped to
// [0, 50] here, so a key repeat cannot walk it outside the range the UI means.
// The controller only holds the number; the loop copies it into the streamer.

#ifndef LNT_CAMERA_CONTROLLER_H
#define LNT_CAMERA_CONTROLLER_H

#include "LNTCamera.h"

namespace lnt {

class LntCameraController {
public:
    static constexpr double kMinDistance = 1.0;
    static constexpr double kMaxDistance = 10000.0;
    static constexpr double kMinSse = 0.0;
    static constexpr double kMaxSse = 50.0;

    LntCameraController();

    // ---- placement -------------------------------------------------------
    void set_target(double x, double y, double z);
    void set_distance(double d);                    // immediate, clamped
    void set_angles(double yaw, double pitch);      // radians, pitch clamped
    void set_lens(double fov_y_deg, double near_clip, double far_clip);

    // ---- actions ---------------------------------------------------------
    void orbit(float delta_yaw, float delta_pitch);     // radians; ignored in 2D
    // 2D plan mode (default): orthographic, looking straight down -Z, orbit locked.
    // Off = perspective 3D orbit. Switching ON snaps back to the plan orientation.
    void set_ortho2d(bool on);
    bool ortho2d() const { return ortho2d_; }
    void pan(float delta_x, float delta_y);             // metres, view right / up
    void zoom(float delta_distance);                    // metres, + = farther
    // Advances the zoom glide. Returns true while still moving.
    bool update(float dt);

    // ---- SSE dial --------------------------------------------------------
    void set_sse_threshold(double px);
    void adjust_sse(double delta_px) { set_sse_threshold(sse_ + delta_px); }
    double sse_threshold() const { return sse_; }

    // ---- state -----------------------------------------------------------
    double yaw() const { return yaw_; }
    double pitch() const { return pitch_; }
    double distance() const { return dist_; }
    double goal_distance() const { return goal_dist_; }
    const double* target() const { return target_; }
    void eye(double out[3]) const;
    double fov_y_deg() const { return fov_; }
    double near_clip() const { return near_; }
    double far_clip() const { return far_; }

    // Fills `cam` (eye, target, up, lens, aspect, height, render origin) and
    // rebuilds its matrices + frustum. Returns false for a degenerate basis.
    bool apply_to(LNTCamera& cam, const double origin[3], double aspect,
                  uint32_t screen_height_px) const;

private:
    double target_[3] = {0.0, 0.0, 0.0};
    double yaw_ = 0.0, pitch_ = 0.0;      // plan view: the default mode is 2D
    double dist_ = 100.0, goal_dist_ = 100.0;
    double fov_ = 45.0, near_ = 1.0, far_ = 100000.0;
    double sse_ = 10.0;
    bool ortho2d_ = true;
};

}  // namespace lnt

#endif  // LNT_CAMERA_CONTROLLER_H
