#pragma once

#include <string_view>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include <Volk/volk.h>

#include "vk/context.hpp"
#include "utility/types.hpp"

#include <array>

namespace whim::vk {

struct allocated_image_2d_t {
  handle<VkImage>     image      = {};
  handle<VkImageView> image_view = {};
  VkExtent2D          extent     = {};
  VkFormat            format     = {};
  VmaAllocation       allocation = nullptr;
};

struct push_constant_t {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

struct buffer_t {
  handle<VkBuffer>      buffer     = nullptr;
  handle<VmaAllocation> allocation = nullptr;
};

struct vertex_t {
  glm::vec3                                                         pos  = { 0.f, 0.f, 0.f };
  glm::vec3                                                         norm = { 0.f, 0.f, 0.f };
  static constexpr VkVertexInputBindingDescription                  description();
  static constexpr std::array<VkVertexInputAttributeDescription, 2> attributes_description();
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

private:
  struct render_frame_data_t {
    // commands
    handle<VkCommandBuffer> command_buffer = {};

    // syncronization
    handle<VkFence>     in_flight_fence           = {};
    handle<VkSemaphore> image_available_semaphore = {};
    handle<VkSemaphore> render_finished_semaphore = {};
  };

  struct imgui_data_t {
    handle<VkDescriptorPool> descriptor_pool = {};
  };

private:
  handle<VkPipeline>       m_pipeline        = {};
  handle<VkPipelineLayout> m_pipeline_layout = {};

  buffer_t m_vertex_buffer = {};
  buffer_t m_index_buffer  = {};

  u32 m_current_frame = 0;

  // NOTE: size == m_frames_count
  std::vector<render_frame_data_t> m_frames_data = {};

  imgui_data_t imgui = {};

  ref<Context> m_context;

private:
  constexpr static u32 m_frames_count = 2;

  constexpr static std::string_view vertex_path   = "./spv/default.vert.spv";
  constexpr static std::string_view fragment_path = "./spv/default.frag.spv";

  VkPipeline     create_pipeline();
};

constexpr VkVertexInputBindingDescription vertex_t::description() {
  VkVertexInputBindingDescription description = {};

  description.binding   = 0;
  description.stride    = sizeof(vertex_t);
  description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  return description;
}

constexpr std::array<VkVertexInputAttributeDescription, 2> vertex_t::attributes_description() {

  std::array<VkVertexInputAttributeDescription, 2> attributes = {};

  attributes[0].binding  = 0;
  attributes[0].location = 0;
  attributes[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
  attributes[0].offset   = offsetof(vertex_t, pos);

  attributes[1].binding  = 0;
  attributes[1].location = 1;
  attributes[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
  attributes[1].offset   = offsetof(vertex_t, norm);

  return attributes;
}
}; // namespace whim::vk
