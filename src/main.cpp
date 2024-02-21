#include "obj_loader.hpp"
#include "vk/renderer.hpp"

#include <Volk/volk.h>

#include <vma/vk_mem_alloc.h>

#include <VkBootstrap.h>

#include <GLFW/glfw3.h>

#include "window.hpp"
#include "whim.hpp"

#include "vk/context.hpp"

int main() {
  WINFO("Hello, world {}", 12);
  WASSERT(true, "THIS SHOULD BE TRUE");
  WERROR("THIS IS ERROR!!{}!", 42);

  config_t config{
    .width = 960, .height = 600, .app_name = "hello world", .options = {.is_resizable = false, .is_fullscreen = false, .raytracing_enabled = false}
  };

  whim::Window       w{ config };
  whim::vk::Context  context{ config, w };
  whim::vk::Renderer renderer{ context };

  // whim::ObjLoader loader("../assets/obj/cube_multi.obj");

  renderer.load_model("../assets/obj/wuson.obj");
  renderer.end_load();

  w.run([&]() {
    if (glfwGetKey(w.handle(), GLFW_KEY_ESCAPE) == GLFW_PRESS) {
      w.close();
    }
    renderer.draw();
  });

  return 0;
}
