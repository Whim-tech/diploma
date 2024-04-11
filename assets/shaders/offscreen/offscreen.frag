#version 460


layout(location = 0) in vec2 out_uv;
layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 0) uniform sampler2D image;

void main() {
    frag_color = texture(image, out_uv);
}