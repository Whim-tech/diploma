#pragma once

#include <string_view>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include <vulkan/vulkan_core.h>

#include "camera.hpp"
#include "shader.h"
#include "vk/context.hpp"
#include "whim.hpp"

namespace whim::vk {

class Renderer {
public:
  explicit Renderer(Context &context, CameraManipulator const &man);
  ~Renderer();

  Renderer(Renderer &&) noexcept            = default;
  Renderer &operator=(Renderer &&) noexcept = default;
  Renderer(const Renderer &)                = delete;
  Renderer &operator=(const Renderer &)     = delete;

  void draw();

  void load_model(std::string_view obj_path, glm::mat4 matrix = glm::mat4{ 1.f });
  void end_load();

private:
  struct render_frame_data_t {
    // commands
    handle<VkCommandBuffer> command_buffer = {};

    // syncronization
    handle<VkFence>     in_flight_fence           = {};
    handle<VkSemaphore> image_available_semaphore = {};
    handle<VkSemaphore> render_finished_semaphore = {};
  };

  struct model_description_t {
    buffer_t index          = {};
    buffer_t vertex         = {};
    buffer_t material       = {};
    buffer_t material_index = {};

    int vertex_count = 0;
    int index_count  = 0;

    glm::mat4 matrix = glm::mat4{ 1.f };
  };

private:
  handle<VkPipeline>       m_pipeline        = {};
  handle<VkPipelineLayout> m_pipeline_layout = {};

  u32 m_current_frame = 0;
  // NOTE: size == m_frames_count
  std::vector<render_frame_data_t> m_frames_data = {};

  handle<VkDescriptorPool> m_imgui_desc_pool = VK_NULL_HANDLE;

  buffer_t                      m_desc_buffer      = {};
  VkDeviceAddress               m_desc_buffer_addr = {};
  std::vector<mesh_description> m_object_desc{};

  std::vector<model_description_t> m_model_desc{};

  ref<Context>            m_context;
  cref<CameraManipulator> m_camera;

  struct {
    handle<VkDescriptorSetLayout> layout = VK_NULL_HANDLE;
    handle<VkDescriptorPool>      pool   = VK_NULL_HANDLE;
    handle<VkDescriptorSet>       set    = VK_NULL_HANDLE;
  } m_desc;

private:
  constexpr static u32 m_frames_count = 2;

  constexpr static std::string_view vertex_path   = "./spv/default.vert.spv";
  constexpr static std::string_view fragment_path = "./spv/default.frag.spv";
};
}; // namespace whim::vk
