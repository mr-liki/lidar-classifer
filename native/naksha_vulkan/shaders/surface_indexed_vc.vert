#version 450
layout(location=0) in vec3 position;
layout(location=1) in vec4 color;
layout(location=0) out vec3 vertexColor;
layout(location=1) out float worldZ;
layout(set=0,binding=0) uniform FrameUbo {
    mat4 mvp; vec4 cameraPos; vec4 origin;
} frame;
void main() {
    gl_Position = frame.mvp * vec4(position, 1.0);
    vertexColor = color.rgb;
    worldZ = position.z + frame.origin.z;
    gl_PointSize = 1.0;
}
