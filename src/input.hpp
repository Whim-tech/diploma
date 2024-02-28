#pragma once

#include "utility/types.hpp"
#include "window.hpp"

namespace whim {

class Input {

public:
  struct state_t {

    struct keyboard_t {
      bool forward_button = false;
      bool back_button    = false;
      bool left_button    = false;
      bool right_button   = false;
      bool up_button      = false;
      bool down_button    = false;

      bool ctrl  = false;
      bool shift = false;
      bool alt   = false;

      bool esc = false;
    } keyboard;

    struct mouse_t {
      bool left_mouse_button   = false;
      bool right_mouse_button  = false;
      bool middle_mouse_button = false;

      f32 mouse_dx = 0.f, mouse_dy = 0.f;

    } mouse;

    struct window_t {

      bool resized = false;
    } window;

    f32 dt = 0.f;
  };

public:
  explicit Input(Window const &window);
  void update();

  void reset();

  state_t const &state() const;

private:
  state_t      m_state = {};
  cref<Window> m_window;

  struct mouse_t {
    f64 x = 0., y = 0.;
  } m_mouse = {};

  f64 m_current_time = 0.;

  constexpr static int forward = GLFW_KEY_W;
  constexpr static int left    = GLFW_KEY_A;
  constexpr static int back    = GLFW_KEY_S;
  constexpr static int right   = GLFW_KEY_D;
};
}; // namespace whim
