#pragma once

#include <vector>
#include <string_view>

#include <GLFW/glfw3.h>

#define VK_NO_PROTOTYPES
#include <vk_mem_alloc.h>
#include <Volk/volk.h>

#include "utility/types.hpp"
#include "window.hpp"
#include "config.hpp"

namespace whim::vk {

class Context {

public:
  // TODO: add options to specify which device we want to choose
  Context(config_t const &config, Window const &window);
  // TODO: add more nice way for COntext creation:
  // for example: context_builder.add_deivice_extention(...)
  //                .add_instance_extension(...)
  //                .enable_instance_layer(...)
  ~Context();

  Context(Context &&) noexcept            = default;
  Context &operator=(Context &&) noexcept = default;
  Context(const Context &)                = delete;
  Context &operator=(const Context &)     = delete;

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

  [[nodiscard]] VkSwapchainKHR           swapchain() const;
  [[nodiscard]] std::vector<VkImage>     swapchain_images() const;
  [[nodiscard]] std::vector<VkImageView> swapchain_image_views() const;
  [[nodiscard]] VkPresentModeKHR         swapchain_present_mode() const;
  [[nodiscard]] VkFormat                 swapchain_image_format() const;
  [[nodiscard]] VkExtent2D               swapchain_extent() const;
  [[nodiscard]] u32                      swapchain_image_count() const;

  [[nodiscard]] GLFWwindow* window();

  // TODO: handling resizing
  //  - recreate swapchain
  // void on_resize(VkExtent2D new_frame_size);

  // TODO: select another physical_device
  //  - update m_device structure
  //  - update vkCommandPool
  //  - update m_swapchain structure
  // void on_changing_device(VkPhysicalDevice new_device, Window const& window);

  void set_debug_name(VkImage image, std::string_view name);
  void set_debug_name(VkImageView image_view, std::string_view name);
  void set_debug_name(VkCommandPool command_pool, std::string_view name);
  void set_debug_name(VkCommandBuffer command_buffer, std::string_view name);
  void set_debug_name(VkFramebuffer frame_buffer, std::string_view name);
  void set_debug_name(VkFence fence, std::string_view name);
  void set_debug_name(VkSemaphore semaphore, std::string_view name);
  void set_debug_name(VkPipeline pipeline, std::string_view name);

private:
  handle<VkInstance>               m_instance        = VK_NULL_HANDLE;
  handle<VkDebugUtilsMessengerEXT> m_debug_messenger = VK_NULL_HANDLE;
  handle<VkSurfaceKHR>             m_surface         = VK_NULL_HANDLE;

  struct device_t {
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

  struct swapchain_t {
    handle<VkSwapchainKHR>   handle       = VK_NULL_HANDLE;
    std::vector<VkImage>     images       = {};
    std::vector<VkImageView> image_views  = {};
    VkPresentModeKHR         present_mode = {};
    VkFormat                 image_format = {};
    VkExtent2D               extent       = {};
    u32                      image_count  = 0;
  } m_swapchain;

  cref<Window> m_window_ref;
};

} // namespace whim::vk
