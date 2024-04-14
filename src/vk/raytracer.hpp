#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan_core.h>
#include <optional>

#include "camera.hpp"
#include "vk/context.hpp"
#include "shader.h"

#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"

#include "vk/types.hpp"
#include "whim.hpp"

namespace whim::vk {
struct primitive_full_info {
  u32 index_count    = 0;
  u32 index_offset   = 0;
  u32 vertex_count   = 0;
  u32 vertex_offset  = 0;
  u32 material_index = 0;
};

struct node {
  glm::mat4 world_matrix   = glm::mat4{ 1.f };
  int       primitive_mesh = 0;
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

  void load_gltf_scene(std::string_view file_path);
  // void load_spheres(std::vector<std::pair<sphere_t, u32>> &spheres, std::vector<material_options> &materials);

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

  void load_gltf_raw(std::string_view file_path);
  void process_node(const tinygltf::Model &tmodel, int &node_idx, const glm::mat4 &parent_matrix);
  void load_gltf_device();
  void load_primitive_to_blas(primitive_full_info &primitive, acceleration_structure_t &blas);

  void create_tlas();
  void init_descriptors();
  void create_pipeline();
  void create_shader_binding_table();

  void update_uniform_buffer(VkCommandBuffer cmd);

  texture_t create_texture(
      u32 width, u32 height,            //
      std::vector<unsigned char> &data, //
      VkFilter mag_filter, VkFilter min_filter, VkFormat format = VK_FORMAT_R8G8B8A8_SRGB
  );

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
  struct {
    struct {
      std::vector<glm::vec3>             positions{};
      std::vector<u32>                   indices{};
      std::vector<glm::vec3>             normals{};
      std::vector<glm::vec2>             uvs{};
      std::vector<material>              materials{};
      std::vector<primitive_shader_info> prim_meshes{};
      //
      std::vector<primitive_full_info>          primitive_infos{};
      std::vector<node>                         nodes{};
      std::unordered_map<i32, std::vector<u32>> mesh_to_primitives{};
    } raw;

    struct {
      buffer_t pos_buffer      = {};
      buffer_t index_buffer    = {};
      buffer_t normal_buffer   = {};
      buffer_t uv_buffer       = {};
      buffer_t material_buffer = {};
      buffer_t prim_infos      = {};
    } device;

    std::vector<acceleration_structure_t> blases{};
  } m_meshes;

  struct {
    std::vector<scene_description> data    = {};
    buffer_t                       buffer  = {};
    VkDeviceAddress                address = {};
  } m_description;

  // SPHERES DATA
  // struct {
  //   struct raw_data {
  //     std::vector<sphere_t>    spheres{};
  //     std::vector<aabb_t>      aabbs{};
  //     std::vector<material>    materials{};
  //     std::vector<u32>         mat_indices{};
  //     std::vector<std::string> texture_names{};
  //   } raw;
  //   struct {
  //     buffer_t spheres        = {};
  //     buffer_t aabbs          = {};
  //     buffer_t material       = {};
  //     buffer_t material_index = {};
  //     u32      spheres_total  = 0;
  //   } gpu_data;
  //   struct blas_data {
  //     handle<VkAccelerationStructureKHR> handle = VK_NULL_HANDLE;
  //     buffer_t                           buffer = {};
  //   } blas;
  //   u32 desc_index = 0;
  // } m_spheres;

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