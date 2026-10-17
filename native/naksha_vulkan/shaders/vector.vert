#version 450
// ---------------------------------------------------------------------------
// Vector overlay (SNT CAD linework) vertex stage.
//
// World-space geometry, exactly like the point and surface paths: the vertex
// buffer holds float32 (world - renderOrigin) in the SAME 12-byte position
// slot, so the shared frame MVP (set 0 binding 0, written by nkv_set_camera_*)
// places the lines without any new camera API.
//
// Vertex layout is the interleaved vector arena:
//   binding 0, stride 16 -> location 0 vec3 position (offset 0)
//                           location 1 vec4 colour   (offset 12, RGBA8)
//
// Colour comes from the CAD entity (already resolved on the Python side to the
// SNT layer/entity colour), never from a palette lookup, and is passed flat to
// the fragment stage so the authored byte survives to the framebuffer.
// ---------------------------------------------------------------------------
layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;

layout(location = 0) out vec3 vertexColor;

layout(set = 0, binding = 0) uniform FrameUbo {
    mat4 mvp;
    vec4 cameraPos;
    vec4 origin;
} frame;

void main() {
    gl_Position = frame.mvp * vec4(position, 1.0);
    vertexColor = color.rgb;
}
