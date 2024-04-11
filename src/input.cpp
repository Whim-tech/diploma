#include "input.hpp"
#include "GLFW/glfw3.h"

namespace whim {

Input::Input(Window const &window) :
    m_window(window) {
  glfwGetCursorPos(m_window.get().handle(), &m_mouse.x, &m_mouse.y);
}

// TODO:
void Input::update() {

  // keyboard
  m_state.keyboard.forward_button = glfwGetKey(m_window.get().handle(), forward) == GLFW_PRESS;
  m_state.keyboard.back_button    = glfwGetKey(m_window.get().handle(), back) == GLFW_PRESS;
  m_state.keyboard.left_button    = glfwGetKey(m_window.get().handle(), left) == GLFW_PRESS;
  m_state.keyboard.right_button   = glfwGetKey(m_window.get().handle(), right) == GLFW_PRESS;
  m_state.keyboard.up_button      = glfwGetKey(m_window.get().handle(), GLFW_KEY_Q) == GLFW_PRESS;
  m_state.keyboard.down_button    = glfwGetKey(m_window.get().handle(), GLFW_KEY_E) == GLFW_PRESS;
  m_state.keyboard.ctrl           = glfwGetKey(m_window.get().handle(), GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS;
  m_state.keyboard.shift          = glfwGetKey(m_window.get().handle(), GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS;
  m_state.keyboard.alt            = glfwGetKey(m_window.get().handle(), GLFW_KEY_LEFT_ALT) == GLFW_PRESS;
  m_state.keyboard.esc            = glfwGetKey(m_window.get().handle(), GLFW_KEY_ESCAPE) == GLFW_PRESS;

  m_state.keyboard.r = glfwGetKey(m_window.get().handle(), GLFW_KEY_R) == GLFW_PRESS;

  m_state.mouse.left_mouse_button   = glfwGetMouseButton(m_window.get().handle(), GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
  m_state.mouse.right_mouse_button  = glfwGetMouseButton(m_window.get().handle(), GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
  m_state.mouse.middle_mouse_button = glfwGetMouseButton(m_window.get().handle(), GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;

  // mouse
  f64 new_mouse_x = 0., new_mouse_y = 0.;
  glfwGetCursorPos(m_window.get().handle(), &new_mouse_x, &new_mouse_y);

  auto extent            = m_window.get().window_size();
  m_state.mouse.mouse_dx = ((f32) new_mouse_x - (f32) m_mouse.x) / extent.width;
  m_state.mouse.mouse_dy = ((f32) new_mouse_y - (f32) m_mouse.y) / extent.height;

  m_mouse.x = new_mouse_x;
  m_mouse.y = new_mouse_y;

  // time
  double now     = glfwGetTime();
  m_state.dt     = (f32) now - (f32) m_current_time;
  m_current_time = now;

  // TODO: screen events
}

void Input::reset() { m_state = state_t{}; }

Input::state_t const &Input::state() const { return m_state; }

} // namespace whim