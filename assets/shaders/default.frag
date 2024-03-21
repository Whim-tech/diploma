#version 450

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : enable

#include "shader.h"

// clang-format off
layout(location = 0) in vec3 vertex_color;

layout(location = 0) out vec4 frag_color;

layout(push_constant) uniform constants { push_constant_t pc; };

layout(buffer_reference, scalar) buffer Descriptions { object_description d[]; };
layout(buffer_reference, scalar) buffer Verteces { vertex v[]; };
layout(buffer_reference, scalar) buffer VertIndices { int i[]; };
layout(buffer_reference) buffer Materials { material m[]; };
layout(buffer_reference, scalar) buffer MatIndices { int i[]; };

// clang-format on

void main() {
  frag_color                          = vec4(vertex_color, 1.f);
  Descriptions       obj_descriptions = Descriptions(pc.obj_address);
  object_description desc             = obj_descriptions.d[pc.obj_index];

  Materials   materials    = Materials(desc.material_address);
  MatIndices  material_ids = MatIndices(desc.material_index_address);
  Verteces    verteces     = Verteces(desc.vertex_address);
  VertIndices vert_indices = VertIndices(desc.index_address);

  int      mat_index = material_ids.i[gl_PrimitiveID];
  material mat       = materials.m[mat_index];
  int      index     = vert_indices.i[gl_PrimitiveID];
  vertex   v         = verteces.v[index];

  frag_color = vec4(v.normal, 1.f);
  // frag_color = vec4(mat.diffuse, 1.f);

}
