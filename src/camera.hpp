#pragma once

#include "glm/glm.hpp"
#include "input.hpp"
#include "utility/types.hpp"

namespace whim {

struct camera_t {

  // view
  glm::vec3 up     = glm::vec3{ 0.f, 1.f, 0.f };
  glm::vec3 center = glm::vec3{ 0.f, 0.f, 0.f };
  glm::vec3 eye    = glm::vec3{ 0.f, 0.f, -3.f };
  // proj
  f32       fov         = 60.f;
  f32       aspect      = 1.f;
  glm::vec2 clip_planes = glm::vec2{ 0.001f, 100.f };
};

class CameraManipulator {
public:
  enum class Mode { Walk };

public:
  CameraManipulator(Input const &input, camera_t camera);

  camera_t const& camera() const;
  camera_t& camera();

  void update();

  void set_look_at(glm::vec3 up, glm::vec3 center, glm::vec3 eye, f32 fov);

  glm::mat4 const &view_matrix() const;
  glm::mat4 const &proj_matrix() const;

  glm::mat4 const &inverse_view_matrix() const;
  glm::mat4 const &inverse_proj_matrix() const;

private:
  void update_view();
  void update_proj();

private:
  camera_t m_camera;

  glm::mat4 m_view_matrix         = glm::mat4{ 1.f };
  glm::mat4 m_proj_matrix         = glm::mat4{ 1.f };
  glm::mat4 m_inverse_view_matrix = glm::mat4{ 1.f };
  glm::mat4 m_inverse_proj_matrix = glm::mat4{ 1.f };

  f32 m_speed = 30.f;

  cref<Input> m_input;
};
} // namespace whim