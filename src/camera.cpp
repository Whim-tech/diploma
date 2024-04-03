#include "camera.hpp"
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/ext/quaternion_geometric.hpp"
#include "glm/matrix.hpp"

namespace whim {

CameraManipulator::CameraManipulator(Input const &input, camera_t camera) :
    m_camera(camera),
    m_input(input) {

  update_proj();
  update_view();
}

camera_t const &CameraManipulator::camera() const { return m_camera; }

camera_t &CameraManipulator::camera() { return m_camera; }

void CameraManipulator::set_look_at(glm::vec3 up, glm::vec3 center, glm::vec3 eye, f32 fov) {
  m_camera.up     = up;
  m_camera.center = center;
  m_camera.eye    = eye;
  m_camera.fov    = fov;
}

// TODO: move input from constructor to update function?...
void CameraManipulator::update() {

  Input::state_t const &state    = m_input.get().state();
  auto                  keyboard = state.keyboard;
  auto                  mouse    = state.mouse;
  auto                  dt       = state.dt;

  glm::vec3 origin        = m_camera.eye;
  glm::vec3 position      = m_camera.center;
  glm::vec3 center_to_eye = position - origin;
  f32       length        = center_to_eye.length();
  center_to_eye           = glm::normalize(center_to_eye);
  // forward direction
  glm::vec3 axe_z = center_to_eye;
  // right direction
  glm::vec3 axe_x = glm::normalize(glm::cross(m_camera.up, axe_z));

  if (keyboard.forward_button) {
    m_camera.eye += axe_z * m_speed * dt;
    m_camera.center += axe_z * m_speed * dt;
  }
  if (keyboard.back_button) {
    m_camera.eye -= axe_z * m_speed * dt;
    m_camera.center -= axe_z * m_speed * dt;
  }
  if (keyboard.right_button) {
    m_camera.eye += axe_x * m_speed * dt;
    m_camera.center += axe_x * m_speed * dt;
  }
  if (keyboard.left_button) {
    m_camera.eye -= axe_x * m_speed * dt;
    m_camera.center -= axe_x * m_speed * dt;
  }

  if ((mouse.mouse_dx != 0 or mouse.mouse_dy != 0) and mouse.right_mouse_button) {
    // Full width will do a full turn
    f32 dx = mouse.mouse_dx * glm::two_pi<f32>();
    f32 dy = mouse.mouse_dy * glm::two_pi<f32>();

    // apply y rotation
    glm::mat4 rot_y = glm::rotate(glm::mat4{ 1.f }, dx, m_camera.up);
    center_to_eye   = rot_y * glm::vec4(center_to_eye, 0.f);

    // applying x rotation
    glm::mat4 rot_x    = glm::rotate(glm::mat4{ 1.f }, dy, axe_x);
    glm::vec3 vect_rot = rot_x * glm::vec4(center_to_eye, 0.f);
    if (glm::sign(vect_rot.x) == glm::sign(center_to_eye.x)) {
      center_to_eye = vect_rot;
    }

    center_to_eye *= length;

    glm::vec3 new_pos = center_to_eye + origin;

    m_camera.center = new_pos;
  }

  update_view();
}

glm::mat4 const &CameraManipulator::view_matrix() const { return m_view_matrix; }

glm::mat4 const &CameraManipulator::proj_matrix() const { return m_proj_matrix; }

glm::mat4 const &CameraManipulator::inverse_view_matrix() const { return m_inverse_view_matrix; }

glm::mat4 const &CameraManipulator::inverse_proj_matrix() const { return m_inverse_proj_matrix; }

// TODO: maybe there is easier way to calculate inverse matrices
void CameraManipulator::update_view() {
  m_view_matrix         = glm::lookAt(m_camera.eye, m_camera.center, m_camera.up);
  m_inverse_view_matrix = glm::inverse(m_view_matrix);
}

// TODO: maybe there is easier way to calculate inverse matrices
void CameraManipulator::update_proj() {
  m_proj_matrix         = glm::perspective(m_camera.fov / 2.f, m_camera.aspect, m_camera.clip_planes.x, m_camera.clip_planes.y);
  m_inverse_proj_matrix = glm::inverse(m_proj_matrix);
}
}; // namespace whim