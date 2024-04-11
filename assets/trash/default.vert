#version 450

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : enable

#include "shader.h"

// clang-format off
layout(location = 0) in vec3 i_position;
layout(location = 1) in float i_ux;
layout(location = 2) in vec3 i_normal;
layout(location = 3) in float i_uy;

layout(location = 0) out vec3 o_color;

layout(push_constant) uniform constants { push_constant_t pc; };

// clang-format on

void main() {

  o_color = i_normal;
  gl_Position = pc.mvp * vec4(i_position, 1.f);

}
