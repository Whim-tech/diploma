#include "camera.hpp"
#include "input.hpp"
#include "obj_loader.hpp"
#include "vk/renderer.hpp"

#include <Volk/volk.h>

#include <vma/vk_mem_alloc.h>

#include <VkBootstrap.h>

#include <GLFW/glfw3.h>

#include "window.hpp"
#include "whim.hpp"

#include "vk/context.hpp"
#include "utility/types.hpp"

int main() {
  WINFO("Hello, world {}", 12);
  WASSERT(true, "THIS SHOULD BE TRUE");
  WERROR("THIS IS ERROR!!{}!", 42);

  config_t config{
    .width    = 960,
    .height   = 600,
    .app_name = "hello world", //
    .options  = {//
      .is_resizable       = false, //
      .is_fullscreen      = false, //
      .raytracing_enabled = false
      }
  };

  whim::Window   w{ config };
  whim::Input    input{ w };
  whim::camera_t camera{
    .aspect = (whim::f32) config.width / (whim::f32) config.height,
  };

  whim::CameraManipulator cam_man{ input, camera };

  whim::vk::Context  context{ config, w };
  whim::vk::Renderer renderer{ context, cam_man };

  // whim::ObjLoader loader("../assets/obj/cube_multi.obj");

  renderer.load_model("../assets/obj/cube_multi.obj");
  renderer.load_model("../assets/obj/wuson.obj");
  renderer.load_model("../assets/obj/sponza.obj");
  renderer.end_load();

  w.run([&]() {
    input.update();

    if (input.state().mouse.right_mouse_button) {
      w.disable_cursor();
    } else {
      w.enable_cursor();
    }

    cam_man.update();

    if (input.state().keyboard.esc) {
      w.close();
    }

    renderer.draw();

    input.reset();
  });

  return 0;
}
