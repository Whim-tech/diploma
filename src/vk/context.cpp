#include "vk/context.hpp"

#include <stdexcept>
#include <Volk/volk.h>

#include <VkBootstrap.h>
#include <vulkan/vulkan_core.h>
#include "whim.hpp"
#include "vk/result.hpp"

namespace whim::vk {

Context::Context(config_t const &config, Window const &window) :
    m_window_ref(window) {
  WINFO("starting Context initialization");

  volkInitialize();
  vkb::InstanceBuilder instance_builder;

  auto required_extensions = window.get_vulkan_required_extensions();

  // TODO: add custom debug messenger
  auto inst_result = instance_builder //
                         .require_api_version(VK_API_VERSION_1_3)
                         .set_app_version(0, 1, 0)
                         .set_app_name(config.app_name.c_str())
                         .set_engine_name("WHIM ENGINE")
                         .enable_validation_layers(config.options.validation_layers_support)
                         .enable_extension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)
                         .enable_extensions(required_extensions)
                         .use_default_debug_messenger()
                         .enable_layer("VK_LAYER_LUNARG_monitor")
                         .build();

  if (!inst_result.has_value()) {
    WERROR("Failed to get VulkanINstance, message: {}", inst_result.error().message());
    throw std::runtime_error("failed to get VkInstance");
  }
  m_instance        = inst_result->instance;
  m_debug_messenger = inst_result->debug_messenger;
  volkLoadInstance(m_instance);
  WINFO("created Vulkan Instance {}", (void*) m_instance);

  m_surface = window.create_surface(inst_result->instance);

  vkb::PhysicalDeviceSelector selector{ inst_result.value() };

  selector = selector //
                 .set_surface(m_surface)
                 .add_required_extension(VK_KHR_SWAPCHAIN_EXTENSION_NAME)
                 .set_minimum_version(1, 3);

  if (config.options.raytracing_enabled) {
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_feature = {};
    accel_feature.sType                                            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_feature = {};
    rt_pipeline_feature.sType                                         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;

    selector = selector //
                   .add_required_extension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)
                   .add_required_extension_features(rt_pipeline_feature)
                   .add_required_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
                   .add_required_extension_features(accel_feature)
                   .add_required_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
  }
  auto ph_device_result = selector.select();

  if (!ph_device_result.has_value()) {
    WERROR("Failed to select suitable physical device, message: {}", ph_device_result.error().message());
    throw std::runtime_error("Failed to select physical device");
  }

  m_device.physical = ph_device_result->physical_device;
  WINFO("physical device selected: {}", ph_device_result->properties.deviceName);

  // WINFO("extensions:");
  // for (auto extension : ph_device_result->get_available_extensions()) {
  //   WINFO("{}", extension);
  // }

  vkb::DeviceBuilder device_builder{ ph_device_result.value() };
  auto               device_result = device_builder.build();

  if (!device_result.has_value()) {
    WERROR("Failed to create Vulkan Logical Device, message: {}", device_result.error().message());
    throw std::runtime_error("Failed to create VkDevice");
  }

  m_device.logical = device_result->device;
  WINFO("created logical device {}", (void*) m_device.logical);

  volkLoadDevice(m_device.logical);

  // TODO: add error handling (but its kinda should never fail)
  auto graphic                   = device_result->get_queue_index(vkb::QueueType::graphics);
  m_device.graphics_family_index = graphic.value();
  m_device.graphics_queue        = device_result->get_queue(vkb::QueueType::graphics).value();

  auto compute                  = device_result->get_queue_index(vkb::QueueType::compute);
  m_device.compute_family_index = compute.value();
  m_device.compute_queue        = device_result->get_queue(vkb::QueueType::compute).value();

  auto present                  = device_result->get_queue_index(vkb::QueueType::present);
  m_device.present_family_index = present.value();
  m_device.present_queue        = device_result->get_queue(vkb::QueueType::present).value();

  VmaVulkanFunctions vulkan_functions    = {};
  vulkan_functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
  vulkan_functions.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;

  VmaAllocatorCreateInfo alloc_create_info = {};
  alloc_create_info.vulkanApiVersion       = VK_API_VERSION_1_2;
  alloc_create_info.instance               = m_instance;
  alloc_create_info.physicalDevice         = m_device.physical;
  alloc_create_info.device                 = m_device.logical;
  alloc_create_info.pVulkanFunctions       = &vulkan_functions;

  check(
      vmaCreateAllocator(&alloc_create_info, &m_vma), //
      "creating VMA Allocator"
  );
  WINFO("Initialized VMA allocator");

  // TODO: remove command pool from Context
  VkCommandPoolCreateInfo create_info = {};
  create_info.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  create_info.pNext                   = nullptr;
  create_info.queueFamilyIndex        = m_device.graphics_family_index;
  // TODO: add this as a option to constructor
  bool allow_reset  = true;
  create_info.flags = allow_reset ? VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT : 0;

  check(
      vkCreateCommandPool(m_device.logical, &create_info, nullptr, &m_command_pool), //
      "creating command pool"
  );
  WINFO("created main command pool");

  vkb::SwapchainBuilder swapchain_builder{ device_result.value() };

  auto window_size = window.window_size();

  // TODO: add this as a option to a constructor
  VkPresentModeKHR present_mode     = VK_PRESENT_MODE_FIFO_KHR;
  auto             swapchain_result = swapchain_builder //
                              .set_desired_extent(window_size.width, window_size.height)
                              .set_desired_present_mode(present_mode)
                              // FIXME: some imgui issue
                              // also external\imgui\src\imgui_impl_vulkan.cpp:1501
                              .set_desired_format({ VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
                              .build();

  if (!swapchain_result.has_value()) {
    WERROR("Failed to create swapchain, message: {}", swapchain_result.error().message());
    throw std::runtime_error("Failed to create swapchain");
  }
  WINFO("created Vulkan Swapchain");

  m_swapchain.handle       = swapchain_result->swapchain;
  m_swapchain.images       = swapchain_result->get_images().value();
  m_swapchain.image_views  = swapchain_result->get_image_views().value();
  m_swapchain.present_mode = swapchain_result->present_mode;
  m_swapchain.image_format = swapchain_result->image_format;
  m_swapchain.extent       = swapchain_result->extent;
  m_swapchain.image_count  = swapchain_result->image_count;

  set_debug_name(m_command_pool, "main command_pool");
  for (u32 i = 0; i < m_swapchain.image_count; i += 1) {
    set_debug_name(m_swapchain.images[i], fmt::format("swapchain_image #{}", i + 1));
    set_debug_name(m_swapchain.image_views[i], fmt::format("swapchain_image_view #{}", i + 1));
  }
}

Context::~Context() {
  // check if context is valid in case if it is moved
  if (m_instance != nullptr) {
    WINFO("Destroying vulkan context");
    /*
      ORDER OF DESTRUCTION:
        1. waiting until device is done touching our images
        2. destroying of swapchain's iamge_views
        3. destroying of swapchain
        4. destroying of command_pool
        5. destroying of device
        6. destroying of debug message utils
        7. destroying of instance
    */

    vkDeviceWaitIdle(m_device.logical);

    for (auto image_view : m_swapchain.image_views) {
      vkDestroyImageView(m_device.logical, image_view, nullptr);
    }

    vkDestroySwapchainKHR(m_device.logical, m_swapchain.handle, nullptr);
    vkDestroyCommandPool(m_device.logical, m_command_pool, nullptr);
    vmaDestroyAllocator(m_vma);

    vkDestroyDevice(m_device.logical, nullptr);
    vkDestroySurfaceKHR(m_instance, m_surface, nullptr);

    if (m_debug_messenger != nullptr) {
      vkDestroyDebugUtilsMessengerEXT(m_instance, m_debug_messenger, nullptr);
    }

    vkDestroyInstance(m_instance, nullptr);
    // invalidate context after destructions
    m_instance        = nullptr;
    m_debug_messenger = nullptr;
    m_surface         = nullptr;
    m_device          = {};
    m_vma             = nullptr;
    m_command_pool    = {};
    m_swapchain       = {};
  }
}

void Context::set_debug_name(VkImage image, std::string_view name) {

  VkDebugUtilsObjectNameInfoEXT info = {};
  info.sType                         = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  info.objectType                    = VK_OBJECT_TYPE_IMAGE;
  info.objectHandle                  = (uint64_t) image;
  info.pObjectName                   = name.data();

  check(
      vkSetDebugUtilsObjectNameEXT(m_device.logical, &info), //
      fmt::format("setting name:{} to VkImage:{}", name, fmt::ptr(image))
  );
}

void Context::set_debug_name(VkImageView image_view, std::string_view name) {

  VkDebugUtilsObjectNameInfoEXT info = {};
  info.sType                         = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  info.objectType                    = VK_OBJECT_TYPE_IMAGE_VIEW;
  info.objectHandle                  = (uint64_t) image_view;
  info.pObjectName                   = name.data();

  check(
      vkSetDebugUtilsObjectNameEXT(m_device.logical, &info), //
      fmt::format("setting name:{} to VkImageView:{}", name, fmt::ptr(image_view))
  );
}

void Context::set_debug_name(VkCommandPool command_pool, std::string_view name) {

  VkDebugUtilsObjectNameInfoEXT info = {};
  info.sType                         = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  info.objectType                    = VK_OBJECT_TYPE_COMMAND_POOL;
  info.objectHandle                  = (uint64_t) command_pool;
  info.pObjectName                   = name.data();

  check(
      vkSetDebugUtilsObjectNameEXT(m_device.logical, &info), //
      fmt::format("setting name:{} to VkCommandPool:{}", name, fmt::ptr(command_pool))
  );
}

void Context::set_debug_name(VkCommandBuffer command_buffer, std::string_view name) {

  VkDebugUtilsObjectNameInfoEXT info = {};
  info.sType                         = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  info.objectType                    = VK_OBJECT_TYPE_COMMAND_BUFFER;
  info.objectHandle                  = (uint64_t) command_buffer;
  info.pObjectName                   = name.data();

  check(
      vkSetDebugUtilsObjectNameEXT(m_device.logical, &info), //
      fmt::format("setting name:{} to VkCommandBuffer:{}", name, fmt::ptr(command_buffer))
  );
}

void Context::set_debug_name(VkFramebuffer frame_buffer, std::string_view name) {

  VkDebugUtilsObjectNameInfoEXT info = {};
  info.sType                         = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  info.objectType                    = VK_OBJECT_TYPE_FRAMEBUFFER;
  info.objectHandle                  = (uint64_t) frame_buffer;
  info.pObjectName                   = name.data();

  check(
      vkSetDebugUtilsObjectNameEXT(m_device.logical, &info), //
      fmt::format("setting name:{} to VkFramebuffer:{}", name, fmt::ptr(frame_buffer))
  );
}

void Context::set_debug_name(VkFence fence, std::string_view name) {

  VkDebugUtilsObjectNameInfoEXT info = {};
  info.sType                         = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  info.objectType                    = VK_OBJECT_TYPE_FENCE;
  info.objectHandle                  = (uint64_t) fence;
  info.pObjectName                   = name.data();

  check(
      vkSetDebugUtilsObjectNameEXT(m_device.logical, &info), //
      fmt::format("setting name:{} to VkFence:{}", name, fmt::ptr(fence))
  );
}

void Context::set_debug_name(VkSemaphore semaphore, std::string_view name) {

  VkDebugUtilsObjectNameInfoEXT info = {};
  info.sType                         = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  info.objectType                    = VK_OBJECT_TYPE_SEMAPHORE;
  info.objectHandle                  = (uint64_t) semaphore;
  info.pObjectName                   = name.data();

  check(
      vkSetDebugUtilsObjectNameEXT(m_device.logical, &info), //
      fmt::format("setting name:{} to VkSemaphore:{}", name, fmt::ptr(semaphore))
  );
}

void Context::set_debug_name(VkPipeline pipeline, std::string_view name) {

  VkDebugUtilsObjectNameInfoEXT info = {};
  info.sType                         = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  info.objectType                    = VK_OBJECT_TYPE_PIPELINE;
  info.objectHandle                  = (uint64_t) pipeline;
  info.pObjectName                   = name.data();

  check(
      vkSetDebugUtilsObjectNameEXT(m_device.logical, &info), //
      fmt::format("setting name:{} to VkPipeline:{}", name, fmt::ptr(pipeline))
  );
}

[[nodiscard]] VkInstance Context::instance() const { return m_instance; }

[[nodiscard]] VkSurfaceKHR Context::surface() const { return m_surface; }

[[nodiscard]] VkPhysicalDevice Context::physical_device() const { return m_device.physical; }

[[nodiscard]] VkDevice Context::device() const { return m_device.logical; }

[[nodiscard]] VkQueue Context::graphics_queue() const { return m_device.graphics_queue; }

[[nodiscard]] VkQueue Context::compute_queue() const { return m_device.compute_queue; }

[[nodiscard]] VkQueue Context::present_queue() const { return m_device.present_queue; }

[[nodiscard]] u32 Context::graphics_family_index() const { return m_device.graphics_family_index; }

[[nodiscard]] u32 Context::compute_family_index() const { return m_device.compute_family_index; }

[[nodiscard]] u32 Context::present_family_index() const { return m_device.present_family_index; }

[[nodiscard]] VkCommandPool Context::command_pool() const { return m_command_pool; }

[[nodiscard]] VmaAllocator Context::vma_allocator() const { return m_vma; }

[[nodiscard]] VkSwapchainKHR Context::swapchain() const { return m_swapchain.handle; }

[[nodiscard]] std::vector<VkImage> Context::swapchain_images() const { return m_swapchain.images; }

[[nodiscard]] std::vector<VkImageView> Context::swapchain_image_views() const { return m_swapchain.image_views; }

[[nodiscard]] VkPresentModeKHR Context::swapchain_present_mode() const { return m_swapchain.present_mode; }

[[nodiscard]] VkFormat Context::swapchain_image_format() const { return m_swapchain.image_format; }

[[nodiscard]] VkExtent2D Context::swapchain_extent() const { return m_swapchain.extent; }

[[nodiscard]] u32 Context::swapchain_image_count() const { return m_swapchain.image_count; }

[[nodiscard]] GLFWwindow* Context::window() { return m_window_ref.get().handle(); }
} // namespace whim::vk
