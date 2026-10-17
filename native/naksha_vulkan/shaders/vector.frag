#version 450
// ---------------------------------------------------------------------------
// Vector overlay fragment stage.
//
// The swapchain is VK_FORMAT_B8G8R8A8_SRGB, so the hardware encodes every
// written value linear -> sRGB. The CAD colour is an authored 8-bit byte, so
// the same sRGB -> linear pre-compensation gui/... surface_indexed_vc.frag uses
// is applied here: the byte the SNT layer/entity colour resolved to is what
// lands in the framebuffer, instead of being re-encoded to a brighter value.
//
// Lines are opaque (alpha is dropped) so overlapping linework does not
// accumulate a different shade than a single stroke.
// ---------------------------------------------------------------------------
layout(location = 0) in vec3 vertexColor;
layout(location = 0) out vec4 outColor;

void main() {
    vec3 rgb = vertexColor;
    vec3 linearRGB = mix(rgb / 12.92,
                         pow((rgb + 0.055) / 1.055, vec3(2.4)),
                         greaterThan(rgb, vec3(0.04045)));
    outColor = vec4(linearRGB, 1.0);
}
