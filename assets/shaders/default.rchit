#version 460

#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "shader.h"
#include "ray_common.glsl"

// clang-format off
hitAttributeEXT vec2 attribs;

layout(location = 0) rayPayloadInEXT hitPayload prd;

layout(set = 0, binding = TLAS) uniform accelerationStructureEXT top_level_as;
layout(set = 0, binding = UniformBuffer) uniform _GlobalUniforms { global_ubo ubo; };
layout(set = 0, binding = ObjectDescriptions, scalar) buffer Descriptions { mesh_description d[]; } scene_desc;
layout(set = 0, binding = Textures) uniform sampler2D textureSamplers[];
layout(push_constant) uniform constants { push_constant_t pc; };

layout(buffer_reference, scalar) buffer Verteces { vertex v[]; };
layout(buffer_reference, scalar) buffer VertIndices { uvec3 i[]; };
layout(buffer_reference, scalar) buffer Materials { material m[]; };
layout(buffer_reference, scalar) buffer MatIndices { uint i[]; };

// clang-format on

void main() {

  mesh_description desc        = scene_desc.d[gl_InstanceCustomIndexEXT];
  Verteces         verteces    = Verteces(desc.vertex_address);
  VertIndices      indices     = VertIndices(desc.index_address);
  Materials        materials   = Materials(desc.material_address);
  MatIndices       mat_indices = MatIndices(desc.material_index_address);

  uvec3      ind      = indices.i[gl_PrimitiveID];
  vertex     v0       = verteces.v[ind.x];
  vertex     v1       = verteces.v[ind.y];
  vertex     v2       = verteces.v[ind.z];
  const vec3 b        = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
  const vec3 pos      = v0.pos * b.x + v1.pos * b.y + v2.pos * b.z;
  const vec3 worldPos = vec3(gl_ObjectToWorldEXT * vec4(pos, 1.0)); // Transforming the position to world space

  // Computing the normal at hit position
  const vec3 normal      = v0.normal * b.x + v1.normal * b.y + v2.normal * b.z;
  const vec3 worldNormal = normalize(vec3(normal * gl_WorldToObjectEXT)); // Transforming the normal to world space
  // prd.hitValue = abs(worldNormal);

  // vec2 text_coord = v0.texture * b.x + v1.texture * b.y + v2.texture * b.z;
  // prd.hitValue    = texture(textureSamplers[nonuniformEXT(desc.txt_offset)], text_coord).xyz;
  uint     material_index = mat_indices.i[gl_PrimitiveID];
  material mat            = materials.m[material_index];

  vec2 text_coord = v0.texture * b.x + v1.texture * b.y + v2.texture * b.z;
  if (mat.texture_id >= 0) {
    uint text_index = mat.texture_id + desc.txt_offset;
    prd.hitValue    = texture(textureSamplers[text_index], text_coord).xyz;
  } else {
    prd.hitValue = texture(textureSamplers[0], text_coord).xyz;
  }
}
