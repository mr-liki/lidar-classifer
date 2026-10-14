// LNTShadingModes.h - MicroStation-style point cloud presentation modes for the
// native .lnt pipeline. No VTK anywhere: the colour is computed on the GPU.
//
//   RGB            the file's own 8-bit colour
//   Intensity      stored intensity, scaled by [intensity_min, intensity_max], through a ramp
//   Elevation      world Z over [elevation_min, elevation_max], through a ramp
//   Classification the .ptc class palette (unchanged)
//   NormalShading  N.L hillshade from depth-reconstructed normals + Eye-Dome Lighting
//
// HOW IT RUNS. One ShadingUniforms goes to the engine (nkv_set_shading_mode). The engine
// turns it into: two 256-entry ramp tables in the frame UBO, the elevation and intensity
// ranges in the point push constants, the display mode, and - for NormalShading - the
// light and the EDL strength in the lighting pass. Nothing else changes, so a switch does
// not touch a vertex buffer: the resident .lnt tiles stay exactly where they are.
//
// evaluate_ramp() below is the CPU reference of the engine's ramp. The verification suite
// renders a frame and checks every coloured pixel against it, so the GPU and this header
// cannot drift apart.

#ifndef LNT_SHADING_MODES_H
#define LNT_SHADING_MODES_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace lnt {

enum class LNTDisplayMode : uint32_t {
    RGB = 0,
    Intensity = 1,
    Elevation = 2,
    Classification = 3,
    NormalShading = 4
};

enum class LNTColorRamp : uint32_t {
    Grayscale = 0,
    Hue = 1,          // blue (low) -> green -> red (high)
    BlueRed = 2,
    LandTopo = 3
};

#pragma pack(push, 1)
struct ShadingUniforms {
    uint32_t display_mode;     // LNTDisplayMode
    uint32_t color_ramp;       // LNTColorRamp
    float elevation_min;       // Z-start for the elevation ramp (world units)
    float elevation_max;       // Z-end for the elevation ramp
    float intensity_min;       // intensity scale min, fraction 0..1 of the stored range
    float intensity_max;       // intensity scale max, fraction 0..1
    float light_direction[3];  // direction the light travels (NormalShading)
    float edl_strength;        // Eye-Dome Lighting strength, 0 = off (NormalShading)
};
#pragma pack(pop)

static_assert(sizeof(ShadingUniforms) == 40, "ShadingUniforms layout mismatch");
static_assert(offsetof(ShadingUniforms, display_mode) == 0, "display_mode offset");
static_assert(offsetof(ShadingUniforms, color_ramp) == 4, "color_ramp offset");
static_assert(offsetof(ShadingUniforms, elevation_min) == 8, "elevation_min offset");
static_assert(offsetof(ShadingUniforms, elevation_max) == 12, "elevation_max offset");
static_assert(offsetof(ShadingUniforms, intensity_min) == 16, "intensity_min offset");
static_assert(offsetof(ShadingUniforms, intensity_max) == 20, "intensity_max offset");
static_assert(offsetof(ShadingUniforms, light_direction) == 24, "light_direction offset");
static_assert(offsetof(ShadingUniforms, edl_strength) == 36, "edl_strength offset");

// ---- reference ramps -------------------------------------------------------------
// t in [0,1] -> rgb in [0,1]. Hue is the HSV hue circle from 240 deg (blue) down to 0 deg
// (red): the usual GLSL hue() expression with the hue running backwards, so the lowest and
// highest values are different colours (the unreversed circle returns to red at t = 1, which
// makes an elevation ramp ambiguous).
inline void evaluate_ramp(LNTColorRamp ramp, float t, float out[3]) {
    t = std::min(std::max(t, 0.0f), 1.0f);
    switch (ramp) {
        case LNTColorRamp::Hue: {
            const float h6 = (1.0f - t) * 4.0f;
            const float off[3] = {0.0f, 4.0f, 2.0f};
            for (int i = 0; i < 3; ++i) {
                const float m = std::fmod(h6 + off[i], 6.0f);
                out[i] = std::min(std::max(std::fabs(m - 3.0f) - 1.0f, 0.0f), 1.0f);
            }
            break;
        }
        case LNTColorRamp::BlueRed:
            out[0] = t; out[1] = 0.0f; out[2] = 1.0f - t;
            break;
        case LNTColorRamp::LandTopo: {
            const float a[3] = {0.1f, 0.5f, 0.1f}, b[3] = {0.9f, 0.8f, 0.5f};
            for (int i = 0; i < 3; ++i) out[i] = a[i] + (b[i] - a[i]) * t;
            break;
        }
        default:
            out[0] = out[1] = out[2] = t;
            break;
    }
}

// The byte the engine writes for ramp entry i of 256 (what the framebuffer holds).
inline void ramp_entry_bytes(LNTColorRamp ramp, int i, uint8_t out[3]) {
    float rgb[3];
    evaluate_ramp(ramp, float(i) / 255.0f, rgb);
    for (int k = 0; k < 3; ++k) out[k] = static_cast<uint8_t>(rgb[k] * 255.0f + 0.5f);
}

// ---- ready-made configurations ---------------------------------------------------------
inline ShadingUniforms make_shading(LNTDisplayMode mode, LNTColorRamp ramp = LNTColorRamp::Grayscale,
                                    float z_min = 0.0f, float z_max = 1.0f,
                                    float i_min = 0.0f, float i_max = 1.0f) {
    ShadingUniforms u{};
    u.display_mode = static_cast<uint32_t>(mode);
    u.color_ramp = static_cast<uint32_t>(ramp);
    u.elevation_min = z_min;
    u.elevation_max = z_max;
    u.intensity_min = i_min;
    u.intensity_max = i_max;
    // sun from the north-west, 45 degrees up, travelling toward the south-east and down
    u.light_direction[0] = 0.5f;
    u.light_direction[1] = -0.5f;
    u.light_direction[2] = -0.7071f;
    u.edl_strength = 0.0f;
    return u;
}

// ---- crisp shaded-classification surface lighting ---------------------------------------
// The exact model used by the VTK shaded-classification view
// (gui/shading_display.py) and by the engine's surface.frag modes 4/5. The native .lnt
// Shading mesh is lit by the SAME engine shaders, so this header only has to carry the
// parameters and reproduce the derivation of the push constants (azimuth, sharpness,
// ambient, key, fill) that the Python side computes in
// _configure_microstation_color_blend_lighting(). Every constant here is transcribed
// from gui/shading_display.py so the two renderers cannot drift; native/lnt/main.cpp
// checks this header against the closed-form model.

inline constexpr float kShadingSharpnessLegacyMax = 90.0f;   // _SHADING_SHARPNESS_LEGACY_MAX
inline constexpr float kShadingSharpnessMax = 999.0f;         // _SHADING_SHARPNESS_MAX
inline constexpr float kShadingSharpnessStrongMax = 200.0f;   // _SHADING_SHARPNESS_STRONG_MAX
inline constexpr float kShadingSharpnessStrongResponse = 0.90f;  // _SHADING_SHARPNESS_STRONG_RESPONSE
inline constexpr float kShadingMinLightElevation = 12.0f;     // _shading_effective_light_elevation floor
inline constexpr float kShadingAmbientFloor = 0.08f;          // NAKSHA_SHADING_BLEND_AMBIENT_FLOOR default
inline constexpr float kShadingDirectionalEnergy = 1.05f;     // NAKSHA_SHADING_DIRECTIONAL_ENERGY default
inline constexpr float kShadingBaseLightElevation = 45.0f;    // _shading_fixed_light_elevation default

struct SurfaceShading {
    float azimuth_deg = 45.0f;           // compass: where the light comes FROM (last_shade_azimuth)
    float sharpness = 45.0f;             // raw 0..999 Sharpness slider (shading_sharpness_angle)
    float ambient = 0.25f;               // user Ambient (shade_ambient), brightness / shadow floor
    float light_elevation_deg = kShadingBaseLightElevation;  // configured sun elevation
    float ambient_floor = kShadingAmbientFloor;              // NAKSHA_SHADING_BLEND_AMBIENT_FLOOR
    float directional_energy = kShadingDirectionalEnergy;    // NAKSHA_SHADING_DIRECTIONAL_ENERGY
};

// _shading_sharpness_response(value) -> (legacy, overdrive)
inline void shading_sharpness_response(float value, float& legacy, float& overdrive) {
    value = std::min(std::max(value, 0.0f), kShadingSharpnessMax);
    legacy = std::min(std::max(value / kShadingSharpnessLegacyMax, 0.0f), 1.0f);
    if (value <= kShadingSharpnessLegacyMax) { overdrive = 0.0f; return; }
    const float strong_span = std::max(kShadingSharpnessStrongMax - kShadingSharpnessLegacyMax, 1.0f);
    if (value <= kShadingSharpnessStrongMax) {
        overdrive = kShadingSharpnessStrongResponse * ((value - kShadingSharpnessLegacyMax) / strong_span);
    } else {
        const float tail_span = std::max(kShadingSharpnessMax - kShadingSharpnessStrongMax, 1.0f);
        overdrive = kShadingSharpnessStrongResponse +
                    (1.0f - kShadingSharpnessStrongResponse) * ((value - kShadingSharpnessStrongMax) / tail_span);
    }
    overdrive = std::min(std::max(overdrive, 0.0f), 1.0f);
}

// _crisp_sharpness_contrast_gain(sharpness_value)
inline float shading_contrast_gain(float legacy, float overdrive) {
    return std::min(std::max(0.55f + 0.90f * legacy + 5.0f * overdrive, 0.35f), 6.50f);
}

// _shading_effective_light_elevation(app, sharpness_overdrive)
inline float shading_effective_light_elevation(float base_elevation_deg, float overdrive) {
    return std::min(std::max(base_elevation_deg -
                             (base_elevation_deg - kShadingMinLightElevation) * overdrive,
                             kShadingMinLightElevation), 85.0f);
}

// The key/fill split _configure_microstation_color_blend_lighting() applies to the VTK
// light rig. Constant total directional energy: Sharpness redistributes it between the key
// light and the opposite fill instead of acting as a brightness control (Ambient owns that).
inline void shading_key_fill(const SurfaceShading& s, float& key_intensity, float& fill_intensity) {
    float legacy = 0.0f, overdrive = 0.0f;
    shading_sharpness_response(s.sharpness, legacy, overdrive);
    float ratio = 0.58f + 0.37f * legacy + 0.049f * overdrive;
    ratio = std::min(std::max(ratio, 0.50f), 0.999f);
    const float total = std::min(std::max(s.directional_energy, 0.25f), 1.50f);
    key_intensity = std::min(std::max(total * ratio, 0.0f), 2.0f);
    fill_intensity = std::min(std::max(total * (1.0f - ratio), 0.0f), 1.0f);
}

// The exact scalar the engine's surface.frag mode 4 (crisp base) writes for a face whose
// unit normal is n with the given lighting parameters. Reference for native/lnt/main.cpp.
inline float crisp_face_shade(const SurfaceShading& s, const double n[3]) {
    float legacy = 0.0f, overdrive = 0.0f;
    shading_sharpness_response(s.sharpness, legacy, overdrive);
    const float gain = shading_contrast_gain(legacy, overdrive);
    const float eff_elev = shading_effective_light_elevation(s.light_elevation_deg, overdrive);
    const float ambient = std::min(std::max(s.ambient, 0.0f), 0.95f);
    const float shadow_floor = std::max(ambient, s.ambient_floor);
    const double kDeg = 3.14159265358979323846 / 180.0;
    const double zenith = (90.0 - eff_elev) * kDeg;
    const double az = (360.0 - s.azimuth_deg + 90.0) * kDeg;
    const double L[3] = { std::sin(zenith) * std::cos(az), std::sin(zenith) * std::sin(az),
                          std::cos(zenith) };
    const float ndl = float(n[0] * L[0] + n[1] * L[1] + n[2] * L[2]);
    const float raw = std::min(std::max(std::max(ndl, ambient), 0.0f), 1.0f);
    const float raw_horizontal = std::max(float(std::sin(eff_elev * kDeg)), ambient);
    const float target_horizontal = std::max(float(std::sin(s.light_elevation_deg * kDeg)), shadow_floor);
    return std::min(std::max(target_horizontal + (raw - raw_horizontal) * gain, shadow_floor), 1.0f);
}

}  // namespace lnt

#endif  // LNT_SHADING_MODES_H
