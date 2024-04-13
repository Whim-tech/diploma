#pragma once

#include <vector>
#include <string_view>

#include <GLFW/glfw3.h>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

#include "window.hpp"
#include "config.hpp"

#include "whim.hpp"

namespace whim::vk {

struct swapchain_frame_t {
  VkImage     image      = VK_NULL_HANDLE;
  VkImageView image_view = VK_NULL_HANDLE;

  struct depth_t {
    VkImage       image      = VK_NULL_HANDLE;
    VkImageView   image_view = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
  } depth;
};

class Context {

public:
  // TODO: add options to specify which device we want to choose
  Context(config_t const &config, Window const &window);
  // TODO: add more nice way for COntext creation:
  // for example: context_builder.add_device_extention(...)
  //                .add_instance_extension(...)
  //                .enable_instance_layer(...)
  ~Context();

  Context(Context &&) noexcept            = default;
  Context &operator=(Context &&) noexcept = default;
  Context(const Context &)                = delete;
  Context &operator=(const Context &)     = delete;

  void immediate_submit(std::function<void(VkCommandBuffer cmd)> &&function) const;

  VkDeviceAddress get_buffer_device_address(VkBuffer buffer) const;

  image_t create_image_on_gpu(VkImageCreateInfo image_info, u8* data, size_t size);

  void generate_mipmaps(VkImage image, VkImageCreateInfo image_info);

  buffer_t create_buffer(
      VkDeviceSize size, const void* data_, //
      VkBufferUsageFlags usage_, VkMemoryPropertyFlags memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
  );

  template<typename T>
  buffer_t create_buffer(
      std::vector<T>    &data, //
      VkBufferUsageFlags usage, VkMemoryPropertyFlags mem_props_ = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
  ) {
    WASSERT(not data.empty(), "data vector should be non empty!");
    return create_buffer(sizeof(T) * data.size(), data.data(), usage, mem_props_);
  }

  VkShaderModule create_shader_module(std::string_view file_path) const;

  void transition_image(
      VkCommandBuffer cmd, VkImage image, //
      VkImageLayout currentLayout, VkImageLayout newLayout
  ) const;

  void transition_image(
      VkCommandBuffer cmd, VkImage image,                   //
      VkImageLayout currentLayout, VkImageLayout newLayout, //
      VkImageSubresourceRange subresource
  ) const;

  [[nodiscard]] VkInstance   instance() const;
  [[nodiscard]] VkSurfaceKHR surface() const;

  [[nodiscard]] VkPhysicalDevice physical_device() const;
  [[nodiscard]] VkDevice         device() const;
  [[nodiscard]] VkQueue          graphics_queue() const;
  [[nodiscard]] VkQueue          compute_queue() const;
  [[nodiscard]] VkQueue          present_queue() const;
  [[nodiscard]] u32              graphics_family_index() const;
  [[nodiscard]] u32              compute_family_index() const;
  [[nodiscard]] u32              present_family_index() const;

  [[nodiscard]] VkCommandPool command_pool() const;
  [[nodiscard]] VmaAllocator  vma_allocator() const;

  [[nodiscard]] VkSwapchainKHR                        swapchain() const;
  [[nodiscard]] std::vector<swapchain_frame_t> const &swapchain_frames() const;
  [[nodiscard]] VkPresentModeKHR                      swapchain_present_mode() const;
  [[nodiscard]] VkFormat                              swapchain_image_format() const;
  [[nodiscard]] VkFormat                              swapchain_depth_format() const;
  [[nodiscard]] VkExtent2D                            swapchain_extent() const;
  [[nodiscard]] u32                                   swapchain_image_count() const;

  [[nodiscard]] GLFWwindow* window() const;

  // TODO: handling resizing
  //  - recreate swapchain
  // void on_resize(VkExtent2D new_frame_size);

  // TODO: select another physical_device
  //  - update m_device structure
  //  - update vkCommandPool
  //  - update m_swapchain structure
  // void on_changing_device(VkPhysicalDevice new_device, Window const& window);

  void set_debug_name(VkImage image, std::string_view name) const;
  void set_debug_name(VkImageView image_view, std::string_view name) const;
  void set_debug_name(VkCommandPool command_pool, std::string_view name) const;
  void set_debug_name(VkCommandBuffer command_buffer, std::string_view name) const;
  void set_debug_name(VkFramebuffer frame_buffer, std::string_view name) const;
  void set_debug_name(VkFence fence, std::string_view name) const;
  void set_debug_name(VkSemaphore semaphore, std::string_view name) const;
  void set_debug_name(VkPipeline pipeline, std::string_view name) const;
  void set_debug_name(VkBuffer buffer, std::string_view name) const;

private:
  handle<VkInstance>               m_instance        = VK_NULL_HANDLE;
  handle<VkDebugUtilsMessengerEXT> m_debug_messenger = VK_NULL_HANDLE;
  handle<VkSurfaceKHR>             m_surface         = VK_NULL_HANDLE;

  struct {
    handle<VkPhysicalDevice> physical              = VK_NULL_HANDLE;
    handle<VkDevice>         logical               = VK_NULL_HANDLE;
    handle<VkQueue>          graphics_queue        = VK_NULL_HANDLE;
    handle<VkQueue>          compute_queue         = VK_NULL_HANDLE;
    handle<VkQueue>          present_queue         = VK_NULL_HANDLE;
    u32                      graphics_family_index = 0;
    u32                      compute_family_index  = 0;
    u32                      present_family_index  = 0;
  } m_device;

  handle<VmaAllocator>  m_vma          = VK_NULL_HANDLE;
  handle<VkCommandPool> m_command_pool = VK_NULL_HANDLE;

  struct {
    handle<VkSwapchainKHR> handle       = VK_NULL_HANDLE;
    VkPresentModeKHR       present_mode = {};
    VkFormat               image_format = {};
    // FIXME: remove this hardcoded format
    VkFormat   depth_format = VK_FORMAT_D32_SFLOAT;
    VkExtent2D extent       = {};
    u32        image_count  = 0;
  } m_swapchain;

  std::vector<swapchain_frame_t> m_frames{};

  struct {
    handle<VkFence>         fence      = VK_NULL_HANDLE;
    handle<VkCommandPool>   cmd_pool   = VK_NULL_HANDLE;
    handle<VkCommandBuffer> cmd_buffer = VK_NULL_HANDLE;
  } m_immediate_data;

  cref<Window> m_window_ref;

private:
};

} // namespace whim::vk
