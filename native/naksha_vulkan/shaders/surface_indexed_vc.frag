#version 450
layout(location=0) in vec3 vertexColor;
layout(location=1) in float worldZ;
layout(push_constant) uniform SurfaceColor {
    layout(offset=16) vec4 elevRangeMode;
    vec4 shadeParamsA;
    vec4 elevationColor;
} pc;
layout(location=0) out vec4 outColor;
void main() {
    vec3 rgb = vertexColor;
    if (pc.elevationColor.x > 0.5) {
        float t = clamp((worldZ-pc.elevationColor.y)/max(pc.elevationColor.z-pc.elevationColor.y,1e-6),0.0,1.0);
        int ramp = int(pc.elevationColor.w+0.5);
        if (ramp == 1) rgb = clamp(abs(mod((1.0-t)*4.0+vec3(0,4,2),6.0)-3.0)-1.0,0.0,1.0);
        else if (ramp == 2) rgb = vec3(t,0,1.0-t);
        else if (ramp == 3) rgb = mix(vec3(0.1,0.5,0.1),vec3(0.9,0.8,0.5),t);
        else rgb = vec3(t);
        rgb *= clamp(max(pc.shadeParamsA.z,sin(radians(pc.elevRangeMode.w))),0.0,1.0);
    }
    vec3 linearRGB = mix(rgb / 12.92, pow((rgb + 0.055) / 1.055, vec3(2.4)), greaterThan(rgb, vec3(0.04045)));
    outColor = vec4(linearRGB, 1.0);
}
