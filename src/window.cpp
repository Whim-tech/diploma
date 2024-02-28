#include "window.hpp"

#include <stdexcept>
#include <utility>

#include "GLFW/glfw3.h"
#include "vk/result.hpp"

namespace whim {
Window::Window(config_t const &config) {

  if (!glfwInit()) {
    WERROR("failed to initialize GLFW");
    throw std::runtime_error("failed to initialize GLFW");
  }

  if (!glfwVulkanSupported()) {
    WERROR("seems like this device cant run vulkan");
    throw std::runtime_error("vulkan is not supported on this device");
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, config.options.is_resizable ? GLFW_TRUE : GLFW_FALSE);

  GLFWmonitor* monitor = config.options.is_fullscreen ? glfwGetPrimaryMonitor() : nullptr;
  m_handle             = glfwCreateWindow(config.width, config.height, config.app_name.c_str(), monitor, nullptr);

  if (m_handle == nullptr) {
    WERROR("Failed to create GLFW window");
    throw std::runtime_error("failed to create glfw window");
  }

}

Window::~Window() {
  if (m_handle != nullptr) {
    glfwDestroyWindow(m_handle);
    m_handle = nullptr;
    glfwTerminate();
  }
}

// FIXME: should fun be passed as reference?...
void Window::run(std::function<void(void)> fun) {
  while (!glfwWindowShouldClose(m_handle)) {

    fun();
    glfwPollEvents();
  }
}

void Window::disable_cursor() { glfwSetInputMode(m_handle, GLFW_CURSOR, GLFW_CURSOR_DISABLED); }

void Window::enable_cursor() { glfwSetInputMode(m_handle, GLFW_CURSOR, GLFW_CURSOR_NORMAL); }

void Window::close() { glfwSetWindowShouldClose(m_handle, GLFW_TRUE); }

VkExtent2D Window::window_size() const {
  int width = 0, height = 0;
  glfwGetWindowSize(m_handle, &width, &height);
  return { (u32) width, (u32) height };
}

std::pair<u32, u32> Window::framebuffer_size() const {
  int width = 0, height = 0;
  glfwGetFramebufferSize(m_handle, &width, &height);
  return { (u32) width, (u32) height };
}

GLFWwindow* Window::handle() const { return m_handle; }

std::vector<const char*> Window::get_vulkan_required_extensions() const {
  uint32_t     extensions_count = 0;
  const char** glfw_extensions  = glfwGetRequiredInstanceExtensions(&extensions_count);
  return std::vector<const char*>(glfw_extensions, glfw_extensions + extensions_count); // NOLINT
}

VkSurfaceKHR Window::create_surface(VkInstance instance) const {
  VkSurfaceKHR surface = nullptr;

  vk::check(                                                          //
      glfwCreateWindowSurface(instance, m_handle, nullptr, &surface), //
      "Creating Vulkan Surface"
  );

  return surface;
}

} // namespace whim
