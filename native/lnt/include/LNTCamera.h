// LNTCamera.h - camera, frustum planes and node culling for the .lnt streamer.
//
// WHY THIS IS NOT THE ENGINE'S CAMERA. native/naksha_vulkan already has three
// camera/frustum implementations (naksha::renderer::Camera with its own
// FrustumPlanes, the FrustumPlanes + ExtractFrustumPlanes + AabbVisible group in
// SurfaceTiling.hpp, and the VTK-parallel wrapper in Camera.hpp). None of them
// can be used here, and the reason is a build constraint rather than taste: the
// whole point of LNTVulkanEngine.h is that the .lnt pipeline compiles on a
// machine with no Vulkan SDK, no MSVC and no naksha headers at all - it reaches
// the engine only through LoadLibrary/GetProcAddress at runtime. Including
// naksha/SurfaceTiling.hpp would pull <cmath>-plus-project-headers into a build
// command that is deliberately `g++ -O3 -std=c++17 main.cpp -o main`.
//
// So the math is re-implemented here, and the re-implementation is not allowed
// to be an approximation: culling with planes that disagree with the matrix the
// GPU actually used removes geometry the frame would have drawn, or keeps
// payloads that are off-screen. Three things keep it honest.
//
// 1. THE PREFERRED PATH TAKES THE ENGINE'S OWN MATRIX. nkv_get_last_mvp() reads
//    the `mvp` field of the frame UBO the renderer filled this frame - the exact
//    bytes the shader multiplies by - and set_from_engine() derives the planes
//    from it. No convention is then assumed anywhere.
//
// 2. THE STANDALONE PATH REPRODUCES THE ENGINE'S CONVENTIONS EXACTLY, and the
//    harness checks that by comparing it against (1) whenever the engine is
//    live. The conventions, read off the engine rather than guessed:
//      * right-handed view space, the camera looks down -Z;
//      * view rows are (right, up, -forward) with the translation in column 4;
//      * projection row y is NEGATED, because Vulkan NDC +y points down the
//        framebuffer while the VTK view being matched treats +y as up;
//      * Vulkan depth is [0, 1];
//      * matrix storage is row-major with the COLUMN-vector convention
//        (p' = M * p), and the GLSL export is the transpose-to-column-major of
//        that - see Renderer/Camera.cpp GetViewProjectionFloat.
//    The world up vector is (0, 1, 0), which is the engine's actual behaviour:
//    naksha::Camera declares `up_ = {0, 1, 0}` and never assigns it, so
//    ApplyToCore() passes it to the core camera on every path even though
//    Camera.hpp's own comment claims Z-up. Copying the header's comment instead
//    of the code would have tilted every reconstructed frustum.
//
// 3. THE CULL TEST IS CONSERVATIVE AT BOTH ENDS, AND ITS SLACK IS PER PLANE AND
//    MEASURED. The box is tested in float64 world space against float64 planes,
//    and a node is discarded only when it lies outside a plane by more than that
//    plane's budget. The budgets are not uniform, because the errors are not:
//    five planes agree with their float64 counterparts to under a millimetre,
//    while the far plane's DISTANCE is only as good as float32 can carry, so it
//    is granted a budget proportional to its own distance (see
//    kFarPlaneToleranceFloor). What they have in common is the direction they err
//    in: both keep geometry the GPU may then clip, and neither ever drops a cell
//    the GPU could have drawn. An unreadable frustum (no engine frame yet, or a
//    degenerate MVP) reports EVERY node visible rather than none: drawing too
//    much is recoverable, dropping real geometry is not.
//
// THE WORKING SPACE. Node cells are world coordinates (UTM-magnitude, up to
// ~6.8e5 m). The planes coming out of the MVP are in RENDER space, because the
// renderer builds its matrices from (world - render origin) - that shift is the
// floating-origin rule and it is why the matrix is usable in float32 at all.
// Rather than subtract a 6.8e5 m origin from every node box (which would throw
// away the float64 work above), the PLANES are moved into world space instead:
//
//     n . p_render + d >= 0,  p_render = p_world - origin
//  => n . p_world  + (d - n . origin) >= 0
//
// which is exact in float64 and costs one dot product per plane.

#ifndef LNT_CAMERA_H
#define LNT_CAMERA_H

#include <cmath>
#include <cstdint>
#include <limits>

#include "LNTFormat.h"
#include "LNTMath.h"
#include "LNTReader.h"
#include "LNTTypes.h"

namespace lnt {

// Six planes, each (a, b, c, d) with the INSIDE half-space on the positive side.
enum LntPlane {
    kPlaneLeft = 0,
    kPlaneRight = 1,
    kPlaneBottom = 2,
    kPlaneTop = 3,
    kPlaneNear = 4,
    kPlaneFar = 5,
    kPlaneCount = 6
};

// A node is culled only when it is outside a plane by more than this. See note 3
// in the file header for the error budget this covers.
static constexpr double kCullToleranceMeters = 1e-3;

// THE FAR PLANE'S SLACK, in metres, and why it is stated as a formula rather
// than a constant. row3 - row2 is the one plane whose extraction is
// ill-conditioned once the matrix has been through float32: the far normal's
// length is n/(fa - n), a difference of two quantities of magnitude ~1 whose
// result is ~1e-5 for any real far/near ratio, so the far DISTANCE it implies
// carries a relative error of about eps32 * fa / n. That is measured, not
// assumed: with this file's own harness - a 1200 m envelope, an oblique camera -
// the float32 far plane sits 1.9 m away from the float64 one at a 3080 m far
// clip, 24.6 m at 6080 m and 70.4 m at 21080 m, i.e. 0.06% to 0.33%, while the
// five well-conditioned planes agree to under 5e-4 m. So the budget is:
//
//     max( kFarPlaneToleranceFloor * far_distance,
//          kFarPlaneToleranceSafety * eps32 * far_distance * (far_distance/near_distance) )
//
// The second term is the analytic bound (kFarPlaneToleranceSafety is the margin
// over it); the first keeps the budget sane when the near clip is so small that
// the analytic term explodes. Both err exclusively in the INCLUSIVE direction: a
// far plane that is a little too far keeps geometry the GPU then clips, while
// one that is a little too near is a section of the cloud that never draws.
static constexpr double kFarPlaneToleranceFloor = 0.02;
static constexpr double kFarPlaneToleranceSafety = 4.0;

// ========================================================================
// 4x4 double matrices, row-major, column-vector convention
// ========================================================================
// Storage is index 4*row + col, so element (r, c) is m[4r + c]. This is the
// same convention as the engine's Matrix4d, which is what makes the standalone
// path comparable with the engine's exported matrix entry by entry.

struct Mat4 {
    double m[16];

    double operator()(int r, int c) const { return m[4 * r + c]; }
    double& operator()(int r, int c) { return m[4 * r + c]; }

    static Mat4 identity() {
        Mat4 r{};
        for (int i = 0; i < 16; ++i) r.m[i] = 0.0;
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0;
        return r;
    }
};

inline Mat4 mat4_mul(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double s = 0.0;
            for (int k = 0; k < 4; ++k) s += a(i, k) * b(k, j);
            r(i, j) = s;
        }
    }
    return r;
}

// The engine's GLSL export layout (element (r, c) at index c*4 + r).
inline void mat4_to_column_major(const Mat4& a, float out16[16]) {
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            out16[c * 4 + r] = static_cast<float>(a(r, c));
}

inline Mat4 mat4_from_column_major(const float in16[16]) {
    Mat4 a{};
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            a(r, c) = static_cast<double>(in16[c * 4 + r]);
    return a;
}

// ========================================================================
// Matrix construction - the engine's formulas, mirrored
// ========================================================================

// Perspective, Vulkan depth [0, 1], right-handed view space, negated y row.
// Byte-for-byte the same shape as naksha::renderer::Camera::UpdateMatrices().
inline Mat4 mat4_perspective(double fov_y_deg, double aspect, double near_clip,
                             double far_clip) {
    const double f = 1.0 / std::tan(fov_y_deg * 3.14159265358979323846 / 360.0);
    const double inv_aspect = 1.0 / (aspect > 1e-9 ? aspect : 1e-9);
    const double n = near_clip > 1e-9 ? near_clip : 1e-9;
    const double fa = far_clip > n ? far_clip : n + 1.0;
    Mat4 p{};
    p.m[0] = f * inv_aspect;  p.m[1] = 0;  p.m[2] = 0;                       p.m[3] = 0;
    p.m[4] = 0;               p.m[5] = -f; p.m[6] = 0;                       p.m[7] = 0;
    p.m[8] = 0;               p.m[9] = 0;  p.m[10] = -fa / (fa - n);         p.m[11] = -(fa * n) / (fa - n);
    p.m[12] = 0;              p.m[13] = 0; p.m[14] = -1;                     p.m[15] = 0;
    return p;
}

// Orthographic mirror of the engine's VTK-parallel box. `half_height` is VTK's
// ParallelScale: HALF the visible world height.
inline Mat4 mat4_orthographic(double half_height, double aspect, double near_clip,
                              double far_clip) {
    const double half_h = (std::fabs(half_height) > 1e-12) ? half_height : 1.0;
    const double half_w = half_h * ((aspect > 1e-9) ? aspect : 1.0);
    const double n = near_clip > 1e-9 ? near_clip : 1e-9;
    const double fa = far_clip > n ? far_clip : n + 1.0;
    Mat4 p{};
    p.m[0] = 1.0 / half_w; p.m[1] = 0;          p.m[2] = 0;               p.m[3] = 0;
    p.m[4] = 0;            p.m[5] = -1.0 / half_h; p.m[6] = 0;            p.m[7] = 0;
    p.m[8] = 0;            p.m[9] = 0;          p.m[10] = -1.0 / (fa - n); p.m[11] = -n / (fa - n);
    p.m[12] = 0;           p.m[13] = 0;         p.m[14] = 0;              p.m[15] = 1.0;
    return p;
}

// Right-handed look-at, Z-up-capable, same row order as the engine's
// ComputeViewMatrix(). Returns false for a degenerate basis (eye == target, or
// a forward parallel to `up`), which the caller reports rather than using.
inline bool mat4_look_at(const double eye[3], const double target[3],
                         const double up[3], Mat4& out) {
    double fx = target[0] - eye[0];
    double fy = target[1] - eye[1];
    double fz = target[2] - eye[2];
    const double flen = std::sqrt(fx * fx + fy * fy + fz * fz);
    if (!(flen > 1e-12)) return false;
    fx /= flen; fy /= flen; fz /= flen;

    double rx = fy * up[2] - fz * up[1];
    double ry = fz * up[0] - fx * up[2];
    double rz = fx * up[1] - fy * up[0];
    const double rlen = std::sqrt(rx * rx + ry * ry + rz * rz);
    if (!(rlen > 1e-12)) return false;   // forward parallel to up: no unique roll
    rx /= rlen; ry /= rlen; rz /= rlen;

    const double ux = ry * fz - rz * fy;
    const double uy = rz * fx - rx * fz;
    const double uz = rx * fy - ry * fx;

    out = Mat4::identity();
    out(0, 0) = rx;  out(0, 1) = ry;  out(0, 2) = rz;
    out(0, 3) = -(rx * eye[0] + ry * eye[1] + rz * eye[2]);
    out(1, 0) = ux;  out(1, 1) = uy;  out(1, 2) = uz;
    out(1, 3) = -(ux * eye[0] + uy * eye[1] + uz * eye[2]);
    out(2, 0) = -fx; out(2, 1) = -fy; out(2, 2) = -fz;
    out(2, 3) = (fx * eye[0] + fy * eye[1] + fz * eye[2]);
    return true;
}

// ========================================================================
// Frustum planes
// ========================================================================

// Gribb-Hartmann on a row-major view-projection matrix.
//
// The five lateral/depth planes are row3 +/- rowI. NEAR IS THE EXCEPTION, and it
// is worth being precise because the engine gets this one wrong in a way that
// happens to be harmless. Vulkan clips z to [0, w], so the near plane is
// z_clip >= 0, i.e. ROW2 alone, and the far plane is w_clip - z_clip >= 0, i.e.
// row3 - row2. The engine's ExtractFrustumPlanes (SurfaceTiling.cpp) emits
// row3 + row2 for near while its own comment says "row2.p >= 0": that is the
// OpenGL [-w, w] near plane, which for a perspective projection is satisfied
// everywhere in front of the eye, so it silently culls nothing. Using the
// correct row2 here can therefore only remove nodes whose points are entirely
// in front of the near clip - geometry the GPU discards anyway - so this is a
// strictly tighter frustum than the engine's and cannot drop anything visible.
//
// THE FAR PLANE IS EXTRACTED LIKE ANY OTHER, BUT IT IS NOT TRUSTED LIKE ONE. Its
// normal length is n/(fa - n) - a difference of two quantities of magnitude ~1
// whose result is ~1e-5 for any real far/near ratio - so a float32 matrix puts
// the far DISTANCE off by ~eps32 * fa / n (measured: 1.9 m at a 3080 m far clip,
// 70 m at 21080 m, against sub-millimetre agreement for the other five planes).
// The plane is still used as extracted, because it is right to that accuracy and
// remodelling it from the near plane is not: that was tried, and the direction it
// imposes is easy to get backwards, which puts the far plane BEHIND the eye and
// culls the entire scene. What is granted instead is slack - a measured budget
// the plane may be wrong by in the only direction that is safe, computed by this
// function and reported through `far_tolerance_out`.
//
// Returns false when the matrix is degenerate (all-zero MVP from a frame that
// has not rendered yet, or a projection with a zero-length plane normal).
//
// `far_distance_out` and `far_tolerance_out` (both optional) report the far
// plane's distance from the eye (the near-to-far span) and that budget. Both are
// recorded here because neither can be recovered afterwards: the caller moves
// these planes into world space, where a plane's d is a UTM-sized number rather
// than a distance along the view axis.
inline bool frustum_from_mvp(const Mat4& vp, double planes[kPlaneCount][4],
                            double* far_distance_out = nullptr,
                            double* far_tolerance_out = nullptr) {
    if (far_distance_out) *far_distance_out = 0.0;
    if (far_tolerance_out) *far_tolerance_out = 0.0;
    for (int i = 0; i < kPlaneCount; ++i)
        for (int k = 0; k < 4; ++k) planes[i][k] = 0.0;

    for (int k = 0; k < 4; ++k) {
        const double r0 = vp(0, k), r1 = vp(1, k), r2 = vp(2, k), r3 = vp(3, k);
        planes[kPlaneLeft][k]   = r3 + r0;
        planes[kPlaneRight][k]  = r3 - r0;
        planes[kPlaneBottom][k] = r3 + r1;
        planes[kPlaneTop][k]    = r3 - r1;
        planes[kPlaneNear][k]   = r2;
        planes[kPlaneFar][k]    = r3 - r2;
    }

    // The RAW normal lengths of the near and far planes, taken before anything
    // is normalised. Their ratio is n/fa for every projection this file mirrors:
    // it is the amplification the far plane's extraction suffers, so it is worth
    // twice what it costs (two square roots) to state the far plane's budget
    // from the matrix that actually produced it.
    const double near_raw = std::sqrt(planes[kPlaneNear][0] * planes[kPlaneNear][0] +
                                      planes[kPlaneNear][1] * planes[kPlaneNear][1] +
                                      planes[kPlaneNear][2] * planes[kPlaneNear][2]);
    const double far_raw = std::sqrt(planes[kPlaneFar][0] * planes[kPlaneFar][0] +
                                     planes[kPlaneFar][1] * planes[kPlaneFar][1] +
                                     planes[kPlaneFar][2] * planes[kPlaneFar][2]);

    for (int i = 0; i < kPlaneCount; ++i) {
        const double len = std::sqrt(planes[i][0] * planes[i][0] +
                                     planes[i][1] * planes[i][1] +
                                     planes[i][2] * planes[i][2]);
        if (!(len > 1e-12)) return false;
        const double inv = 1.0 / len;
        for (int k = 0; k < 4; ++k) planes[i][k] *= inv;
    }
    // Anti-parallel unit normals (true for every projection this file mirrors):
    // the gap between the two planes is d_far + d_near along the shared axis, and
    // that gap IS the near-to-far span.
    const double gap = std::fabs(planes[kPlaneFar][3] + planes[kPlaneNear][3]);
    if (far_distance_out) *far_distance_out = gap;
    if (far_tolerance_out) {
        double slack = kFarPlaneToleranceFloor * gap;
        if (near_raw > 0.0 && far_raw > 0.0) {
            const double amplification = near_raw / far_raw;      // fa / n
            slack = std::max(slack, kFarPlaneToleranceSafety *
                                       double(std::numeric_limits<float>::epsilon()) *
                                       gap * amplification);
        }
        *far_tolerance_out = slack;
    }
    return true;
}

// Moves render-space planes into world space: d -= n . origin. Exact in float64.
inline void frustum_to_world(double planes[kPlaneCount][4], const double origin[3]) {
    for (int i = 0; i < kPlaneCount; ++i) {
        planes[i][3] -= planes[i][0] * origin[0] + planes[i][1] * origin[1] +
                        planes[i][2] * origin[2];
    }
}

// Conservative AABB/plane test: false only when the box is provably outside.
// The positive vertex (the corner furthest along the plane normal) is the vertex
// that decides, exactly as the engine's AabbVisible() does - so a box straddling
// a plane is visible.
//
// `tol` is the slack granted to the five well-conditioned planes and `far_tol`
// to the far plane alone, because the two have different error budgets: a
// negative far_tol means "same as tol", which is what a caller with no far-plane
// uncertainty to declare (and every existing caller) gets.
inline bool planes_vs_box(const double planes[kPlaneCount][4],
                          const double bmin[3], const double bmax[3],
                          double tol = kCullToleranceMeters, double far_tol = -1.0) {
    if (!(far_tol >= 0.0)) far_tol = tol;
    for (int i = 0; i < kPlaneCount; ++i) {
        const double* p = planes[i];
        const double px = (p[0] >= 0.0) ? bmax[0] : bmin[0];
        const double py = (p[1] >= 0.0) ? bmax[1] : bmin[1];
        const double pz = (p[2] >= 0.0) ? bmax[2] : bmin[2];
        const double dist = p[0] * px + p[1] * py + p[2] * pz + p[3];
        if (dist < -((i == kPlaneFar) ? far_tol : tol)) return false;
    }
    return true;
}

// ========================================================================
// LNTCamera
// ========================================================================
struct LNTCamera {
    // ---- world-space view state (metres, float64) ----
    double eye[3]    = {0.0, 0.0, 100.0};
    double target[3] = {0.0, 0.0, 0.0};
    // (0,1,0) because that is what the engine actually uses - see note 2.
    double up[3]     = {0.0, 1.0, 0.0};

    // ---- projection ----
    double fov_y_deg = 45.0;
    double aspect = 16.0 / 9.0;
    double near_clip = 0.1;
    double far_clip = 10000.0;
    bool ortho = false;
    double ortho_half_height = 100.0;

    // ---- render origin: the shift the GPU matrices are built against ----
    double origin[3] = {0.0, 0.0, 0.0};

    // Viewport height in pixels. Screen-space error is per-pixel, so this is a
    // property of the view, not a constant.
    uint32_t screen_height_px = 1080;

    // ---- derived (render-space matrices, world-space planes) ----
    Mat4 view = Mat4::identity();
    Mat4 proj = Mat4::identity();
    Mat4 vp = Mat4::identity();
    double planes[kPlaneCount][4] = {};
    bool frustum_valid = false;
    bool from_engine_mvp = false;
    // Render-space far clip and the slack its extraction is worth, both recorded
    // at extraction (see frustum_from_mvp).
    double far_plane_distance_ = 0.0;
    double far_plane_tolerance_ = kCullToleranceMeters;

    // Builds the matrices from eye/target/fov and extracts WORLD-space planes.
    // Returns false (frustum_valid = false, everything visible) for a degenerate
    // view basis, which is reported rather than silently culling the scene away.
    //
    // THE MATRICES ARE RENDER-SPACE, AND THE PLANES ARE NOT. eye/target are WORLD
    // coordinates, but the view matrix is built from them shifted by -origin,
    // exactly as the engine's ApplyToCore() does. This is not a detail: a view
    // matrix carrying a UTM northing of 3.5e6 m loses all its precision in the
    // float32 export, and - more immediately - a view built in world space would
    // put its frustum planes in world space too, so applying the origin shift to
    // move them "into" world space (as the engine-MVP path requires) would shift
    // them a second time and cull everything.
    //
    // The invariant, therefore, is uniform across both entry points:
    //     view / proj / vp   -> render space   (comparable with the engine's MVP)
    //     planes             -> world space    (comparable with node cells)
    bool rebuild() {
        from_engine_mvp = false;
        const double e[3] = {eye[0] - origin[0], eye[1] - origin[1], eye[2] - origin[2]};
        const double t[3] = {target[0] - origin[0], target[1] - origin[1],
                            target[2] - origin[2]};
        if (!mat4_look_at(e, t, up, view)) {
            frustum_valid = false;
            return false;
        }
        proj = ortho ? mat4_orthographic(ortho_half_height, aspect, near_clip, far_clip)
                     : mat4_perspective(fov_y_deg, aspect, near_clip, far_clip);
        vp = mat4_mul(proj, view);
        frustum_valid = frustum_from_mvp(vp, planes, &far_plane_distance_,
                                        &far_plane_tolerance_);
        if (frustum_valid) frustum_to_world(planes, origin);
        return frustum_valid;
    }

    // The preferred path: the exact matrix the GPU projected this frame with.
    // `mvp16` is the engine's column-major float32 view-projection for
    // position = world - origin, which is what nkv_get_last_mvp() returns.
    // `eye_world` is only needed for the screen-space-error SELECTOR (plane
    // culling needs no camera position); pass nullptr when it is not used, and
    // the eye is left untouched.
    bool set_from_engine_mvp(const float mvp16[16], const double origin_in[3],
                             const double eye_world[3] = nullptr) {
        for (int i = 0; i < 3; ++i) origin[i] = origin_in[i];
        if (eye_world) { eye[0] = eye_world[0]; eye[1] = eye_world[1]; eye[2] = eye_world[2]; }
        vp = mat4_from_column_major(mvp16);
        // Keep view/proj meaningful for reporting: proj is unchanged by the
        // caller and view is recovered only if the caller also supplied eye and
        // target, so they are left as configured. vp is authoritative here.
        from_engine_mvp = true;
        frustum_valid = frustum_from_mvp(vp, planes, &far_plane_distance_,
                                        &far_plane_tolerance_);
        if (frustum_valid) frustum_to_world(planes, origin);
        return frustum_valid;
    }

    // ---- culling ---------------------------------------------------------

    // True when the world-space box is not provably outside the frustum. An
    // invalid frustum reports true, so a frame that has not rendered yet draws
    // everything instead of nothing.
    bool visible(const BoundingBox& world_box) const {
        if (!frustum_valid) return true;
        if (world_box.is_empty()) return false;
        // A target inside this box is on the view axis. At long range the
        // float32 MVP's far-plane subtraction can reject the root even though
        // the target is visibly inside it. Keep that enclosing cell when its
        // target depth lies between the actual near and far clip distances.
        bool encloses_target = true;
        for (int i = 0; i < 3; ++i)
            encloses_target &= target[i] >= world_box.min[i] && target[i] <= world_box.max[i];
        if (encloses_target) {
            const double dx = eye[0] - target[0], dy = eye[1] - target[1],
                         dz = eye[2] - target[2];
            const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
            const double rx = world_box.extent(0), ry = world_box.extent(1),
                         rz = world_box.extent(2);
            const double radius = 0.5 * std::sqrt(rx * rx + ry * ry + rz * rz);
            if (distance > radius && distance >= near_clip && distance <= far_clip)
                return true;
        }
        // Slack for the float32 matrix the GPU projected with. The five
        // well-conditioned planes get the flat, generous term the engine's own
        // matrix has always justified (measured agreement with the float64
        // planes is 5e-4 m over this envelope, so far_clip * 2e-6 is a ~10x
        // margin); the far plane gets its own, much larger, budget because its
        // DISTANCE is the ill-conditioned quantity. Both err only in the
        // direction of keeping more than the GPU will draw.
        const double tolerance = std::max(kCullToleranceMeters, far_clip * 2e-6);
        return planes_vs_box(planes, world_box.min, world_box.max, tolerance,
                             far_plane_tolerance());
    }

    // ---- the far plane's own numbers --------------------------------------

    // Its distance from the eye: the near-to-far span the extraction measured,
    // plus the near clip (0.1 m against thousands, so it is not carried). Read
    // from the MATRIX rather than from far_clip so the number reflects the frame
    // that was actually rendered when the planes came from the engine - that path
    // is the one whose far plane is uncertain.
    double far_plane_distance() const { return far_plane_distance_; }

    // The slack that far plane is worth, in metres, computed by
    // frustum_from_mvp from the matrix that produced it. See
    // kFarPlaneToleranceFloor for the derivation and the measurements.
    double far_plane_tolerance() const {
        return std::max(kCullToleranceMeters, far_plane_tolerance_);
    }

    // The direct directory-driven test: an LNTNodeEntry's cell is its centre
    // (stored as a float32 offset from the file's bounds_min) plus/minus its
    // half_size. No payload read, no Reader needed - which is the point of a
    // node-based format - and it is the same conversion Reader::bounds_of()
    // performs, so a cell tested here and a cell reported there agree.
    //
    // Const, and counter-free on purpose: the selector owns the per-frame
    // statistics, so the camera stays a pure value that can be passed by const
    // reference (which is the signature the streaming entry point wants).
    bool visible(const LNTNodeEntry& node, const double file_origin[3]) const {
        const double c[3] = {file_origin[0] + double(node.center[0]),
                             file_origin[1] + double(node.center[1]),
                             file_origin[2] + double(node.center[2])};
        const BoundingBox cell = BoundingBox::from_center_half(c, double(node.half_size));
        return visible(cell);
    }

    // ---- screen-space error --------------------------------------------
    // The selector's metric, exactly as specified:
    //     SSE = (half_size / distance) * screen_height
    // where `distance` is the eye-to-centre distance clamped to a small floor so
    // a node containing the eye cannot divide by zero. A cube's half_size is the
    // natural "how much geometry am I approximating with" term, and multiplying
    // by the viewport height converts a world-space ratio into pixels.
    //
    // This is the ratio form: it omits the focal-length factor, so its threshold
    // is tied to the FOV it was tuned at. pixels_per_meter_at() below is the
    // focal-length-corrected conversion, and the harness checks that the two
    // agree up to exactly that factor - so the relationship is verified rather
    // than assumed.
    double sse(const BoundingBox& cell) const {
        const double c[3] = {cell.mid(0), cell.mid(1), cell.mid(2)};
        double h = 0.5 * cell.extent(0);
        if (!(h > 0.0)) h = 0.5 * cell.extent(1);
        if (!(h > 0.0)) h = 0.5 * cell.extent(2);
        return sse_of_centre(c, h);
    }

    double sse_of_centre(const double c[3], double half_size) const {
        if (ortho) {
            // Parallel projection: size on screen does not depend on distance.
            // Use the distance a perspective camera of this FOV would need to show
            // the same scale, so one SSE threshold means the same thing in 2D/3D.
            const double fov = (fov_y_deg > 1.0 && fov_y_deg < 179.0) ? fov_y_deg : 45.0;
            const double ps = ortho_half_height > 1e-9 ? ortho_half_height : 1e-9;
            const double d_eq = ps / std::tan(fov * 3.14159265358979323846 / 360.0);
            return (half_size / d_eq) * double(screen_height_px);
        }
        double dx = c[0] - eye[0], dy = c[1] - eye[1], dz = c[2] - eye[2];
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        const double floor_m = 1e-3;
        if (!(dist > floor_m)) dist = floor_m;
        return (half_size / dist) * double(screen_height_px);
    }

    // Pixels per metre at a given depth, with the focal-length term. Mirrors the
    // engine's ProjectedScale (SurfaceTiling.cpp): for a perspective camera the
    // factor is height / (2*tan(fovY/2)) at 1 m, divided by depth; for the ortho
    // box it is height / (2 * parallelScale) and depth-independent.
    double pixels_per_meter_at(double distance_m) const {
        if (ortho) {
            const double span = (ortho_half_height > 0.0)
                                    ? 2.0 * ortho_half_height : 1.0;
            return double(screen_height_px) / span;
        }
        const double fov = (fov_y_deg > 1.0 && fov_y_deg < 179.0) ? fov_y_deg : 45.0;
        const double at1m = double(screen_height_px) /
                            (2.0 * std::tan(fov * 3.14159265358979323846 / 360.0));
        const double d = (distance_m > 1e-9) ? distance_m : 1e-9;
        return at1m / d;
    }

    // The focal factor relating the two metrics:
    //
    //     pixels_per_meter_at(d) * half_size == sse() * focal_scale()
    //
    // It is the FOCAL LENGTH IN PIXELS, normalised by the viewport height:
    //
    //     f_px = height / (2*tan(fovY/2))   =>   f_px / height = 1 / (2*tan(fovY/2))
    //
    // which is the conversion from a world-space size/depth ratio to pixels.
    // The specified SSE formula multiplies that ratio by `height` instead of by
    // `f_px`, so it overstates the true pixel size by 2*tan(fovY/2) = 0.828 at
    // the 45-degree default - i.e. by 21%. That is a property of the metric as
    // specified, not an error introduced here, and it is why the harness asserts
    // the identity above rather than assuming either form is the other.
    //
    // The ortho box has no such constant: its pixels-per-metre figure is
    // depth-independent while sse() is not, so 1.0 is returned and the identity
    // is a perspective statement only.
    double focal_scale() const {
        if (ortho) return 1.0;
        const double fov = (fov_y_deg > 1.0 && fov_y_deg < 179.0) ? fov_y_deg : 45.0;
        return 1.0 / (2.0 * std::tan(fov * 3.14159265358979323846 / 360.0));
    }

    // A world-space plane at index i, for reporting. NaN-free by construction.
    double plane(int i, int k) const { return planes[i][k]; }
};

}  // namespace lnt

#endif  // LNT_CAMERA_H
