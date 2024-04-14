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
layout(set = 0, binding = SceneDescriptions, scalar) buffer Descriptions { scene_description scene; };
layout(set = 0, binding = Primitives) readonly buffer _InstanceInfo {primitive_shader_info prim_info[];};
layout(set = 0, binding = Textures) uniform sampler2D textureSamplers[];
layout(push_constant) uniform constants { push_constant_t pc; };

layout(buffer_reference, scalar) readonly buffer Vertices  { vec3 v[]; };
layout(buffer_reference, scalar) readonly buffer Indices   { uint i[]; };
layout(buffer_reference, scalar) readonly buffer Normals   { vec3 n[]; };
layout(buffer_reference, scalar) readonly buffer TexCoords { vec2 t[]; };
layout(buffer_reference, scalar) readonly buffer Materials { material m[]; };

// clang-format on

void main() {
  // Retrieve the Primitive mesh buffer information
  primitive_shader_info pinfo = prim_info[gl_InstanceCustomIndexEXT];

  // Getting the 'first index' for this mesh (offset of the mesh + offset of the triangle)
  uint index_offset  = pinfo.index_offset + (3 * gl_PrimitiveID);
  uint vertex_offset = pinfo.vertex_offset;          // Vertex offset as defined in glTF
  uint mat_index     = max(0, pinfo.material_index); // material of primitive mesh

  Materials materials = Materials(scene.material_address);
  Vertices  vertices  = Vertices(scene.pos_address);
  Indices   indices   = Indices(scene.index_address);
  Normals   normals   = Normals(scene.normal_address);
  TexCoords texCoords = TexCoords(scene.uv_address);

  // Getting the 3 indices of the triangle (local)
  ivec3 triangle_index = ivec3(indices.i[index_offset + 0], indices.i[index_offset + 1], indices.i[index_offset + 2]);
  triangle_index += ivec3(vertex_offset); // (global)

  const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
  // Vertex of the triangle
  const vec3 pos0           = vertices.v[triangle_index.x];
  const vec3 pos1           = vertices.v[triangle_index.y];
  const vec3 pos2           = vertices.v[triangle_index.z];
  const vec3 position       = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
  const vec3 world_position = vec3(gl_ObjectToWorldEXT * vec4(position, 1.0));

  // Normal
  const vec3 nrm0         = normals.n[triangle_index.x];
  const vec3 nrm1         = normals.n[triangle_index.y];
  const vec3 nrm2         = normals.n[triangle_index.z];
  vec3       normal       = normalize(nrm0 * barycentrics.x + nrm1 * barycentrics.y + nrm2 * barycentrics.z);
  const vec3 world_normal = normalize(vec3(normal * gl_WorldToObjectEXT));
  const vec3 geom_normal  = normalize(cross(pos1 - pos0, pos2 - pos0));

  // TexCoord
  const vec2 uv0       = texCoords.t[triangle_index.x];
  const vec2 uv1       = texCoords.t[triangle_index.y];
  const vec2 uv2       = texCoords.t[triangle_index.z];
  const vec2 texcoord0 = uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;

  // Material of the object
  material mat = materials.m[mat_index];

  if (mat.base_color_texture > -1) {
    uint text_index = mat.base_color_texture;
    // prd.hitValue    = vec4(mat.base_color_factor, 1.f) * texture(textureSamplers[text_index], texcoord0);
    prd.hitValue    = vec4(mat.base_color_factor, 1.f) * textureLod(textureSamplers[text_index], texcoord0, 0.0f);
  } else {
    // prd.hitValue = vec4(abs(world_normal), 1.f);
    prd.hitValue = vec4(mat.base_color_factor, 1.f);
  }

}
