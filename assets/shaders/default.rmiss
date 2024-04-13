#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "shader.h"
#include "ray_common.glsl"

layout(location = 0) rayPayloadInEXT hitPayload prd;

void main() { prd.hitValue = vec4(0.3, 0.3, 0.3, 1.0); }