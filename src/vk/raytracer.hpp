#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan_core.h>
#include <optional>

#include "camera.hpp"
#include "vk/context.hpp"
#include "shader.h"

#include "vk/types.hpp"
#include "whim.hpp"

struct material_options {
  glm::vec3   color        = {};
  std::string texture_name = {};
};

namespace whim::vk {

class Mesh {
public:
  struct raw_data {
    std::vector<vertex>      vertexes{};
    std::vector<u32>         indices{};
    std::vector<material>    materials{};
    std::vector<u32>         mat_indices{};
    std::vector<std::string> texture_names{};
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

  void load_spheres(std::vector<std::pair<sphere_t, u32>> &spheres, std::vector<material_options> &materials);
  void init_scene();

  void reset_frame();

private:
  constexpr static u32              max_frames           = 2;
  constexpr static std::string_view default_texture_path = "../assets/texture/default.png";

  /*
    store per frame data
  */
  struct render_frame_data_t {
    handle<VkCommandBuffer> cmd              = VK_NULL_HANDLE;
    handle<VkFence>         fence            = VK_NULL_HANDLE;
    handle<VkSemaphore>     image_semaphore  = VK_NULL_HANDLE;
    handle<VkSemaphore>     render_semaphore = VK_NULL_HANDLE;
  };

private:
  /*
    init function
  */
  void create_frame_data();
  void init_imgui();
  void create_storage_image();
  void create_uniform_buffer();
  void create_default_texture();
  void create_offscreen_renderer();

  void load_obj_file(std::string_view file_path, Mesh::raw_data &mesh);
  void load_mesh_to_gpu(Mesh &mesh);
  void load_textures(Mesh &mesh);
  void load_mesh_to_blas(Mesh &mesh);

  void create_tlas();
  void init_descriptors();
  void create_pipeline();
  void create_shader_binding_table();

  void update_uniform_buffer(VkCommandBuffer cmd);

  std::optional<texture_t> create_texture(std::string_view file_name);

private:
  // IMGUI DATA
  struct {
    handle<VkDescriptorPool> desc_pool;
  } m_imgui;

  // FRAMES DATA
  std::vector<render_frame_data_t> m_frames;
  u32                              m_current_frame = 0;

  u32 m_shader_frame = 0;
  u32 m_maxFrames    = 100;

  // UNIFORM BUFFER DATA
  buffer_t m_ubo = {};

  // MESHES DATA
  std::unordered_map<std::string, Mesh> m_meshes{};

  struct {
    std::vector<mesh_description> data    = {};
    buffer_t                      buffer  = {};
    VkDeviceAddress               address = {};
  } m_description;

  // SPHERES DATA
  struct {
    struct raw_data {
      std::vector<sphere_t>    spheres{};
      std::vector<aabb_t>      aabbs{};
      std::vector<material>    materials{};
      std::vector<u32>         mat_indices{};
      std::vector<std::string> texture_names{};
    } raw;

    struct {
      buffer_t spheres        = {};
      buffer_t aabbs          = {};
      buffer_t material       = {};
      buffer_t material_index = {};
      u32      spheres_total  = 0;
    } gpu_data;

    struct blas_data {
      handle<VkAccelerationStructureKHR> handle = VK_NULL_HANDLE;
      buffer_t                           buffer = {};
    } blas;

    u32 desc_index = 0;
  } m_spheres;

  // TEXTURES DATA
  std::vector<texture_t> m_textures{};
  texture_t              m_default_texture = {};

  // ACCELERATION STRUCTURE DATA
  std::vector<VkAccelerationStructureInstanceKHR> m_blas_instances{};

  struct {
    buffer_t                           buffer = {};
    handle<VkAccelerationStructureKHR> handle = VK_NULL_HANDLE;
  } m_tlas;

  // SHADER BINDING TABLE DATA
  // TODO: class member?...
  std::vector<VkRayTracingShaderGroupCreateInfoKHR> m_shader_groups{};

  buffer_t                        m_sbtb_buffer = {};
  VkStridedDeviceAddressRegionKHR m_gen_region  = {};
  VkStridedDeviceAddressRegionKHR m_miss_region = {};
  VkStridedDeviceAddressRegionKHR m_hit_region  = {};
  VkStridedDeviceAddressRegionKHR m_call_region = {};

  // DESCRIPTOR SETS DATA
  struct {
    struct {
      handle<VkDescriptorPool>      pool   = VK_NULL_HANDLE;
      handle<VkDescriptorSetLayout> layout = VK_NULL_HANDLE;
      handle<VkDescriptorSet>       set    = VK_NULL_HANDLE;
    } shared;

  } m_descriptor;

  struct {
    handle<VkImage>       image      = VK_NULL_HANDLE;
    handle<VkImageView>   view       = VK_NULL_HANDLE;
    handle<VmaAllocation> allocation = VK_NULL_HANDLE;
    handle<VkSampler>     sampler    = VK_NULL_HANDLE;
    // TODO: handle<VkFormat, VkFormat{}>
    VkFormat    format = {};
    VkImageType type   = {};
    u32         width  = 0;
    u32         height = 0;
  } m_storage_image;

  // PIPELINE DATA
  handle<VkPipeline>       m_pipeline        = VK_NULL_HANDLE;
  handle<VkPipelineLayout> m_pipeline_layout = VK_NULL_HANDLE;

  // RAYTRACING DATA
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rt_prop = {};

  // OFFSCREEN RENDER DATA
  struct {
    handle<VkDescriptorPool>      desc_pool   = VK_NULL_HANDLE;
    handle<VkDescriptorSetLayout> desc_layout = VK_NULL_HANDLE;
    handle<VkDescriptorSet>       desc_set    = VK_NULL_HANDLE;

    handle<VkPipeline>       pipeline        = VK_NULL_HANDLE;
    handle<VkPipelineLayout> pipeline_layout = VK_NULL_HANDLE;
  } m_offscreen;

  // REFERENCES
  ref<Context>            m_context_ref;
  cref<CameraManipulator> m_camera_ref;
};
} // namespace whim::vk