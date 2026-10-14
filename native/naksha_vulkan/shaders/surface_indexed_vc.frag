#version 450
layout(location=0) in vec3 vertexColor;
layout(location=0) out vec4 outColor;
void main() {
    vec3 rgb = vertexColor;
    vec3 linearRGB = mix(rgb / 12.92, pow((rgb + 0.055) / 1.055, vec3(2.4)), greaterThan(rgb, vec3(0.04045)));
    outColor = vec4(linearRGB, 1.0);
}
