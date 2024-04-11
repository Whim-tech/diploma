#include "obj_loader.hpp"

#include <filesystem>
#include <stdexcept>

#include <tiny_obj_loader.h>


namespace whim {

ObjLoader::ObjLoader(std::string_view obj_path) {

  if (!std::filesystem::exists(obj_path)) {
    WERROR("Cant parse obj file: file not found - {}", obj_path);
    throw std::runtime_error("cant parse obj file");
  }

  tinyobj::ObjReader reader;
  reader.ParseFromFile(std::string(obj_path));

  if (!reader.Valid()) {
    WERROR("Cant parse obj file: file is not valid: {}:{}", obj_path, reader.Error());
    throw std::runtime_error("cant parse obj file");
  }

  // Collecting the material in the scene
  for (const auto &mat : reader.GetMaterials()) {
    obj_material material;
    material.ambient       = glm::vec3(mat.ambient[0], mat.ambient[1], mat.ambient[2]);
    material.diffuse       = glm::vec3(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
    material.specular      = glm::vec3(mat.specular[0], mat.specular[1], mat.specular[2]);
    material.emission      = glm::vec3(mat.emission[0], mat.emission[1], mat.emission[2]);
    material.transmittance = glm::vec3(mat.transmittance[0], mat.transmittance[1], mat.transmittance[2]);
    material.dissolve      = mat.dissolve;
    material.ior           = mat.ior;
    material.shininess     = mat.shininess;
    material.illum         = mat.illum;

    // TODO: handle textures

    this->materials.emplace_back(material);
  }
  // If there were none, add a default
  if (this->materials.empty()) this->materials.emplace_back(obj_material{});

  const tinyobj::attrib_t &attrib = reader.GetAttrib();

  for (const auto &shape : reader.GetShapes()) {

    this->vertexes.reserve(shape.mesh.indices.size() + vertexes.size());
    this->indices.reserve(shape.mesh.indices.size() + indices.size());
    this->mat_indices.insert(mat_indices.end(), shape.mesh.material_ids.begin(), shape.mesh.material_ids.end());

    for (const auto &index : shape.mesh.indices) {
      obj_vertex vertex       = {};
      size_t     vertex_index = 3 * index.vertex_index;

      vertex.pos = { attrib.vertices[vertex_index + 0], //
                     attrib.vertices[vertex_index + 1], //
                     attrib.vertices[vertex_index + 2] };

      if (!attrib.normals.empty() && index.normal_index >= 0) {
        size_t normal_index = 3 * index.normal_index;
        vertex.norm         = { attrib.normals[normal_index + 0], //
                                attrib.normals[normal_index + 1], //
                                attrib.normals[normal_index + 2] };
      }

      if (!attrib.texcoords.empty() && index.texcoord_index >= 0) {
        size_t texture_index = 2 * index.texcoord_index;
        vertex.texture       = { attrib.texcoords[texture_index], //
                                 1.0f - attrib.texcoords[texture_index + 1] };
      }

      if (!attrib.colors.empty()) {
        size_t color_index = 3 * index.vertex_index;
        vertex.color       = { attrib.colors[color_index],     //
                               attrib.colors[color_index + 1], //
                               attrib.colors[color_index + 2] };
      }

      this->vertexes.push_back(vertex);
      this->indices.push_back(static_cast<int>(indices.size()));
    }
  }

  // Fixing material indices
  for (auto &mi : this->mat_indices) {
    if (mi < 0 || mi > this->materials.size()) mi = 0;
  }

  // Compute normal when no normal were provided.
  if (attrib.normals.empty()) {
    for (size_t i = 0; i < this->indices.size(); i += 3) {
      obj_vertex &v0 = this->vertexes[this->indices[i + 0]];
      obj_vertex &v1 = this->vertexes[this->indices[i + 1]];
      obj_vertex &v2 = this->vertexes[this->indices[i + 2]];

      glm::vec3 n = glm::normalize(glm::cross((v1.pos - v0.pos), (v2.pos - v0.pos)));
      v0.norm     = n;
      v1.norm     = n;
      v2.norm     = n;
    }
  }
}
} // namespace whim
