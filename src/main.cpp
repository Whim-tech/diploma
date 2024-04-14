
#define TINYGLTF_NO_STB_IMAGE_WRITE

#include "whim.hpp"

#include "camera.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/fwd.hpp"
#include "gltf_loader.hpp"
#include "input.hpp"
#include "shader.h"
#include "vk/raytracer.hpp"

#include <vulkan/vulkan_core.h>
#include <vma/vk_mem_alloc.h>
#include <VkBootstrap.h>

#include <GLFW/glfw3.h>

#include "window.hpp"

#include "vk/context.hpp"
#include "utility/types.hpp"

int main() {
  WINFO("Hello, world {}", 12);
  WASSERT(true, "THIS SHOULD BE TRUE");
  WERROR("THIS IS ERROR!!{}!", 42);

  config_t config{
    .width    = 960,
    .height   = 600,
    .app_name = "_", //
    .options  = {//
      .is_resizable       = false, //
      .is_fullscreen      = false, //
      .raytracing_enabled = true
      }
  };

  whim::Window   w{ config };
  whim::Input    input{ w };
  whim::camera_t camera{
    .aspect = (whim::f32) config.width / (whim::f32) config.height, // F
  };
  camera.center = glm::vec3{ 0.f, 0.f, 0.f };
  camera.eye    = glm::vec3{ 0.f, 0.f, 3.f };

  whim::CameraManipulator cam_man{ input, camera };
  // cam_man.set_look_at(glm::vec3(5, 4, -4), glm::vec3(0, 1, 0), glm::vec3(0, 1, 0), 60.f);

  whim::vk::Context context{ config, w };

  WINFO("CREATING RAYTRACER");
  whim::vk::RayTracer raytracer{ context, cam_man };

  WINFO("MESHES LOADED");
  // raytracer.load_gltf_scene("../assets/gltf/DragonAttenuation/DragonAttenuation.gltf");
  // raytracer.load_gltf_scene("../assets/gltf/VertexColorTest/VertexColorTest.gltf");
  // raytracer.load_gltf_scene("../assets/gltf/Sponza/Sponza.gltf");
  // raytracer.load_gltf_scene("../assets/gltf/DamagedHelmet/DamagedHelmet.gltf");
  // raytracer.load_gltf_scene("../assets/gltf/cornellBox/cornellBox.gltf");
  raytracer.load_gltf_scene("../assets/gltf/FlightHelmet/FlightHelmet.gltf");

  // tests
  // raytracer.load_gltf_scene("../assets/gltf/BoomBoxWithAxes/BoomBoxWithAxes.gltf");
  // raytracer.load_gltf_scene("../assets/gltf/TextureCoordinateTest/TextureCoordinateTest.gltf");

  //
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
    if (input.state().keyboard.r) {
    }
    raytracer.reset_frame();

    // renderer.draw();
    raytracer.draw();

    input.reset();
  });

  return 0;
}
