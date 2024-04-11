#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "shader.h"
#include "ray_common.glsl"

// clang-format off
hitAttributeEXT vec2 attribs;
layout(location = 0) rayPayloadInEXT hitPayload prd;

layout(set = 0, binding = TLAS) uniform accelerationStructureEXT top_level_as;
layout(set = 0, binding = UniformBuffer) uniform _GlobalUniforms { global_ubo ubo; };
layout(set = 0, binding = ObjectDescriptions, scalar) buffer Descriptions { mesh_description d[]; } scene_desc;
layout(set = 0, binding = Spheres, scalar) buffer all_spheres { sphere_t spheres[]; };
layout(set = 0, binding = Textures) uniform sampler2D textureSamplers[];
layout(push_constant) uniform constants { push_constant_t pc; };

layout(buffer_reference, scalar) buffer Materials { material m[]; };
layout(buffer_reference, scalar) buffer MatIndices { uint i[]; };

// clang-format on

void main() {
  mesh_description desc        = scene_desc.d[gl_InstanceCustomIndexEXT];
  Materials        materials   = Materials(desc.material_address);
  MatIndices       mat_indices = MatIndices(desc.material_index_address);

  vec3 world_pos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

  sphere_t instance   = spheres[gl_PrimitiveID];
  vec3     world_norm = normalize(world_pos - instance.center);

  const float PI = 3.1415926f;

  vec3  pos = (world_pos - instance.center) / instance.radius;
  float u   = ((atan(pos.x, pos.z) / PI) + 1.0f) * 0.5f;
  float v   = (asin(pos.y) / PI) + 0.5f;

  // prd.hitValue = abs(world_norm);
  uint     material_index = mat_indices.i[gl_PrimitiveID];
  material mat            = materials.m[material_index];
  uint     text_index     = mat.texture_id + desc.txt_offset;

  prd.hitValue = texture(textureSamplers[text_index], vec2(u, v)).xyz;

  // float k = 2.0;
  // float s = sign(sin(k * u) + sin(k * v));

  // if (s == 1.0) {
  //   prd.hitValue = vec3(1., 0., 0.);
  // } else {
  //   prd.hitValue = vec3(0., 1., 0.);
  // }
}