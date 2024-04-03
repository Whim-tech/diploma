#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "camera.hpp"
#include "glm/fwd.hpp"
#include "vk/context.hpp"
#include "shader.h"

#include "vk/types.hpp"
#include "whim.hpp"

namespace whim::vk {

class Mesh {
public:
  struct raw_data {
    std::vector<vertex>   vertexes{};
    std::vector<u32>      indices{};
    std::vector<material> materials{};
    std::vector<u32>      mat_indices{};
  } raw;

  struct gpu_data {
    u32 vertex_count = 0;
    u32 index_count  = 0;

    buffer_t index          = {};
    buffer_t vertex         = {};
    buffer_t material       = {};
    buffer_t material_index = {};
  } gpu;

  struct blas_data {
    handle<VkAccelerationStructureKHR> handle = VK_NULL_HANDLE;
    buffer_t                           buffer = {};
  } blas;

  // index in array of descriptions
  // see m_descriptions in class below
  size_t description_index = 0;
};

class RayTracer {

public:
  RayTracer(Context &context, CameraManipulator const &man);
  ~RayTracer();

  RayTracer(RayTracer &&) noexcept            = default;
  RayTracer &operator=(RayTracer &&) noexcept = default;
  RayTracer(const RayTracer &)                = delete;
  RayTracer &operator=(const RayTracer &)     = delete;

  void draw();
  // loads data from file to mesh struct
  void load_mesh(std::string_view file_path, std::string_view mesh_name);
  // uploads mesh to a scene
  void load_model(std::string_view mesh_name, glm::mat4 transform = glm::mat4{ 1.f });
  void init_scene();

private:
  constexpr static u32 max_frames = 2;

  /*
    store per frame data
  */
  struct render_frame_data_t {
    handle<VkCommandBuffer> cmd              = VK_NULL_HANDLE;
    handle<VkFence>         fence            = VK_NULL_HANDLE;
    handle<VkSemaphore>     image_semaphore  = VK_NULL_HANDLE;
    handle<VkSemaphore>     render_semaphore = VK_NULL_HANDLE;
  };

  /*
    init function
  */
  void create_frame_data();
  void init_imgui();

private:
  void load_obj_file(std::string_view file_path, Mesh::raw_data &mesh);
  void load_mesh_to_gpu(Mesh &mesh);
  void load_mesh_to_blas(Mesh &mesh);

private:
  struct {
    handle<VkDescriptorPool> desc_pool;
  } m_imgui;

  std::vector<render_frame_data_t> m_frames;

  u32 m_current_frame = 0;

  std::unordered_map<std::string, Mesh> m_meshes{};

  struct {
    std::vector<mesh_description> data{};
    buffer_t                      buffer{};
    VkDeviceAddress               address = {};
  } m_description;

  std::vector<VkAccelerationStructureInstanceKHR> m_blas_instances{};

  ref<Context>            m_context_ref;
  cref<CameraManipulator> m_camera_ref;
};
} // namespace whim::vk