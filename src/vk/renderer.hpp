#pragma once

#include <string_view>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include <Volk/volk.h>

#include "shader_interface.h"
#include "vk/context.hpp"
#include "utility/types.hpp"

#include <array>
#include <vulkan/vulkan_core.h>

namespace whim::vk {

struct model_description_t {
  buffer_t index          = {};
  buffer_t vertex         = {};
  buffer_t material       = {};
  buffer_t material_index = {};

  int vertex_count = 0;
  int index_count  = 0;
  
};

struct model_instance_t {
  glm::mat4 transform = glm::mat4{ 1.f };
  u32       index     = 0;
};

class Renderer {
public:
  explicit Renderer(Context &context);
  ~Renderer();

  Renderer(Renderer &&) noexcept            = default;
  Renderer &operator=(Renderer &&) noexcept = default;
  Renderer(const Renderer &)                = delete;
  Renderer &operator=(const Renderer &)     = delete;

  void draw();

  void load_model(std::string_view obj_path);
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

private:
  handle<VkPipeline>       m_pipeline        = {};
  handle<VkPipelineLayout> m_pipeline_layout = {};

  buffer_t m_vertex_buffer = {};
  buffer_t m_index_buffer  = {};

  u32 m_current_frame = 0;

  // NOTE: size == m_frames_count
  std::vector<render_frame_data_t> m_frames_data = {};

  handle<VkDescriptorPool> m_imgui_desc_pool = VK_NULL_HANDLE;

  buffer_t m_desc_buffer = {};
  VkDeviceAddress m_desc_buffer_addr = {};
  std::vector<object_description> m_object_desc{};

  std::vector<model_description_t> m_model_desc{};

  ref<Context> m_context;

private:
  constexpr static u32 m_frames_count = 2;

  constexpr static std::string_view vertex_path   = "./spv/default.vert.spv";
  constexpr static std::string_view fragment_path = "./spv/default.frag.spv";

  VkPipeline create_pipeline();

  void draw_ui();
};

// constexpr VkVertexInputBindingDescription vertex_t::description() {
//   VkVertexInputBindingDescription description = {};

//   description.binding   = 0;
//   description.stride    = sizeof(vertex_t);
//   description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

//   return description;
// }

// constexpr std::array<VkVertexInputAttributeDescription, 2> vertex_t::attributes_description() {

//   std::array<VkVertexInputAttributeDescription, 2> attributes = {};

//   attributes[0].binding  = 0;
//   attributes[0].location = 0;
//   attributes[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
//   attributes[0].offset   = offsetof(vertex_t, pos);

//   attributes[1].binding  = 0;
//   attributes[1].location = 1;
//   attributes[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
//   attributes[1].offset   = offsetof(vertex_t, norm);

//   return attributes;
// }
}; // namespace whim::vk
