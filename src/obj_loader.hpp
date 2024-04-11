#pragma once

#include <vector>
#include <string_view>

#include <glm/glm.hpp>

#include "whim.hpp"

namespace whim {

struct obj_vertex {
  glm::vec3 pos     = {};
  glm::vec3 norm    = {};
  glm::vec2 texture = {};
  glm::vec3 color   = {};
};

struct obj_material {
  glm::vec3 ambient       = {};
  glm::vec3 diffuse       = {};
  glm::vec3 specular      = {};
  glm::vec3 transmittance = {};
  glm::vec3 emission      = {};
  f32       shininess     = {};
  f32       ior           = {}; // index of refraction
  f32       dissolve      = {}; // 1 == opaque; 0 == fully transparent
  // illumination model (see http://www.fileformat.info/format/material/)
  int illum      = {};
  // TODO: add textures
  int texture_id = -1;
};

class ObjLoader {

  ObjLoader() = default;

public:
  explicit ObjLoader(std::string_view obj_path);

  std::vector<obj_vertex>   vertexes    = {};
  std::vector<u32>          indices     = {};
  std::vector<obj_material> materials   = {};
  std::vector<u32>          mat_indices = {};
};
}; // namespace whim