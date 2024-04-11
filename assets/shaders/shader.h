#ifndef SHADER_INTERFACE_HEADER_GUARD_H
#define SHADER_INTERFACE_HEADER_GUARD_H

#ifdef __cplusplus
#include "glm/glm.hpp"
#include <vulkan/vulkan_core.h>
#include <cstdint>

using vec2 = glm::vec2;
using vec3 = glm::vec3;
using vec4 = glm::vec4;
using mat4 = glm::mat4;
using uint = unsigned int;
#endif

// clang-format off
#ifdef __cplusplus // Descriptor binding helper for C++ and GLSL
 #define START_BINDING(a) enum a {
 #define END_BINDING() }
#else
 #define START_BINDING(a)  const uint
 #define END_BINDING() 
#endif

START_BINDING(SharedBindings)
  TLAS = 0,
  StorageImage = 1,
  UniformBuffer  = 2,  
  ObjectDescriptions = 3,
  Textures = 4,
  Spheres = 5,
  total = 6
END_BINDING();

// clang-format on

struct vertex {
  vec3 pos;
  vec3 normal;
  vec2 texture;
};

struct sphere_t {
  vec3  center;
  float radius;
};

struct aabb_t {
  vec3 min;
  vec3 max;
};

struct material {
  vec3  ambient;
  vec3  diffuse;
  vec3  specular;
  vec3  transmittance;
  vec3  emission;
  float shininess;
  float ior;      // index of refraction
  float dissolve; // 1 == opaque; 0 == fully transparent
  // illumination model (see http://www.fileformat.info/format/material/)
  int illum;
  int texture_id;
};

struct mesh_description {
  int      txt_offset;
  uint64_t vertex_address;
  uint64_t index_address;
  uint64_t material_address;
  uint64_t material_index_address;
};

struct global_ubo {
  mat4 view;
  mat4 proj;
  mat4 inverse_view;
  mat4 inverse_proj;
};

struct push_constant_t {
  mat4 mvp;
  uint frame;
};

#ifdef __cplusplus
#include <array>

constexpr VkVertexInputBindingDescription vertex_description() {
  VkVertexInputBindingDescription description = {};

  description.binding   = 0;
  description.stride    = sizeof(vertex);
  description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  return description;
}

constexpr std::array<VkVertexInputAttributeDescription, 3> vertex_attributes_description() {

  std::array<VkVertexInputAttributeDescription, 3> attributes = {};

  attributes[0].binding  = 0;
  attributes[0].location = 0;
  attributes[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
  attributes[0].offset   = offsetof(vertex, pos);

  attributes[1].binding  = 0;
  attributes[1].location = 2;
  attributes[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
  attributes[1].offset   = offsetof(vertex, normal);

  attributes[2].binding  = 0;
  attributes[2].location = 3;
  attributes[2].format   = VK_FORMAT_R32_SFLOAT;
  attributes[2].offset   = offsetof(vertex, texture);

  return attributes;
}
#endif

#endif
