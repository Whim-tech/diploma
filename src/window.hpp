#pragma once

#include <functional>
#include <vector>

#include <Volk/volk.h>
#include <GLFW/glfw3.h>

#include "config.hpp"
#include "utility/types.hpp"

namespace whim {

class Window {

public:
  explicit Window(config_t const &info);
  ~Window();

  Window(Window &&) noexcept            = default;
  Window &operator=(Window &&) noexcept = default;
  Window(const Window &)                = delete;
  Window &operator=(const Window &)     = delete;

  void run(std::function<void(void)>);
  void close();

  void disable_cursor();
  void enable_cursor();

  // [[nodiscard]] std::pair<u32, u32> window_size() const;
  [[nodiscard]] VkExtent2D          window_size() const;
  [[nodiscard]] std::pair<u32, u32> framebuffer_size() const;
  [[nodiscard]] GLFWwindow*         handle() const;

  [[nodiscard]] std::vector<char const*> get_vulkan_required_extensions() const;
  [[nodiscard]] VkSurfaceKHR             create_surface(VkInstance instance) const;

  void on_resize(u32 new_width, u32 new_height);

private:
  ptr_handle<GLFWwindow*> m_handle = nullptr;
};

} // namespace whim
