#define VOLK_IMPLEMENTATION
#include <Volk/volk.h>

#include <VkBootstrap.h>

#include <GLFW/glfw3.h>

#include "window.hpp"
#include "whim.hpp"

#include "vk/context.hpp"

#define VMA_IMPLEMENTATION
#include "vma/vk_mem_alloc.h"

int main() {
  WINFO("Hello, world {}", 12);
  WASSERT(true, "THIS SHOULD BE TRUE");
  WERROR("THIS IS ERROR!!{}!", 42);

  config_t config {
    .width = 600,
    .height = 480,
    .app_name = "hello world",
    .options = {
      .is_resizable = false,
      .is_fullscreen = false,
    }
  };

  whim::Window      w{ config };
  whim::vk::Context context{ config, w };

  w.run([&]() {
    if (glfwGetKey(w.handle(), GLFW_KEY_ESCAPE) == GLFW_PRESS) {
      w.close();
    }
  });

  return 0;
}
