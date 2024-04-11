#include "camera.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/fwd.hpp"
#include "input.hpp"
#include "obj_loader.hpp"
#include "shader.h"
#include "vk/raytracer.hpp"
#include "vk/renderer.hpp"

#include <vulkan/vulkan_core.h>

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
  camera.eye    = glm::vec3{ 10.f, 10.f, 10.f };

  whim::CameraManipulator cam_man{ input, camera };
  // cam_man.set_look_at(glm::vec3(5, 4, -4), glm::vec3(0, 1, 0), glm::vec3(0, 1, 0), 60.f);

  whim::vk::Context context{ config, w };

  // whim::ObjLoader loader("../assets/obj/cube_multi.obj");

  // whim::vk::Renderer renderer{ context, cam_man };
  // renderer.load_model("../assets/obj/cube_multi.obj");
  // renderer.load_model("../assets/obj/wuson.obj");
  // renderer.load_model("../assets/obj/sponza.obj", glm::scale(glm::mat4{ 1.f }, glm::vec3{ 0.01, 0.01, 0.01 }));
  // renderer.end_load();

  WINFO("CREATING RAYTRACER");
  whim::vk::RayTracer raytracer{ context, cam_man };

  WINFO("LOADING MESHES");
  // raytracer.load_mesh("../assets/obj/cat.obj", "cat");
  // for (int i = 0; i < 10; i += 1) {
  //   for (int j = 0; j < 10; j += 1) {
  //     raytracer.load_model(
  //         "cat",                                                                              //
  //         glm::translate(
  //             glm::rotate(glm::mat4{ 1.f }, glm::radians(-90.f), glm::vec3{ 1.f, 0.f, 0.f }), //
  //             glm::vec3{ 50.f * i, (i + j) * 2 + 50, 40.f * j }
  //         )
  //     );
  //   }
  // }

  // raytracer.load_mesh("../assets/obj/cube_multi.obj", "cube");
  // for (int i = 0; i < 10; i += 1) {
  //   for (int j = 0; j < 10; j += 1) {

  //     raytracer.load_model(
  //         "cube", //

  //         glm::translate(glm::mat4{ 1.f }, glm::vec3{ 2.f * i, (i + j) * 0.2 + 5, 2.f * j })
  //     );
  //   }
  // }

  // raytracer.load_mesh("../assets/obj/sponza.obj", "sponza");
  // raytracer.load_model("sponza", glm::scale(glm::mat4{ 1.f }, glm::vec3{ 1.f / 10.f }));
  // raytracer.load_mesh("../assets/obj/Medieval_building.obj", "Medieval_building");
  // raytracer.load_model("Medieval_building");

  std::vector<std::pair<sphere_t, whim::u32>> spheres{};
  std::vector<material_options>               materials{};

  for (int i = 0; i < 100; i += 1) {
    for (int j = 0; j < 100; j += 1) {

      spheres.emplace_back(
          sphere_t{
              glm::vec3{100.f * i, 300.f, 100.f * j},
              100.f
      },
          0
      );
    }
  }
  materials.emplace_back(material_options{
      glm::vec3{0.f, 0.f, 0.f},
      "DOESNT EXIST"
  });

  raytracer.load_spheres(spheres, materials);

  raytracer.load_mesh("../assets/obj/books.obj", "books");
  raytracer.load_model("books", glm::scale(glm::mat4{ 1.f }, glm::vec3{ 1.f / 10.f }));

  raytracer.init_scene();
  WINFO("MESHES LOADED");

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
