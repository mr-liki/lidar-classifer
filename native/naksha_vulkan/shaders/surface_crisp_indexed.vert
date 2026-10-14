#version 450
// Compact crisp Shaded-Class mesh (see surface.frag, CRISP_INDEXED): one shared
// vertex per point. The fragment shader rebuilds the flat face normal itself, so
// the vertex stage only passes the class colour (interpolated for mixed faces).
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec4 inColor;

layout(set = 0, binding = 0) uniform FrameUbo {
    mat4 mvp;
    vec4 cameraPos;
    vec4 origin;
} ubo;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec4 vColor;
layout(location = 2) out vec3 vLocalPos;
layout(location = 3) flat out uint vTriangleBase;

void main() {
    gl_Position = ubo.mvp * vec4(inPos, 1.0);
    vNormal = vec3(0.0, 0.0, 1.0);
    vColor = inColor;
    vLocalPos = inPos;
    vTriangleBase = uint(gl_InstanceIndex);
}
