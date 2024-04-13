#include "vk/context.hpp"

#include <filesystem>
#include <stdexcept>
#include <fstream>

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include <VkBootstrap.h>
#include <vk_mem_alloc.h>

#include "whim.hpp"
#include "vk/result.hpp"
#include "vk/loader.hpp"

namespace whim::vk {

Context::Context(config_t const &config, Window const &window) :
    m_window_ref(window) {
  WINFO("starting Context initialization");

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
    WERROR("Failed to get VulkanInstance, message: {}", inst_result.error().message());
    throw std::runtime_error("failed to get VkInstance");
  }

  m_instance        = inst_result->instance;
  m_debug_messenger = inst_result->debug_messenger;
  WINFO("created Vulkan Instance {}", (void*) m_instance);

  m_surface = window.create_surface(inst_result->instance);

  vkb::PhysicalDeviceSelector selector{ inst_result.value() };

  selector = selector //
                 .set_surface(m_surface)
                 .add_required_extension(VK_KHR_SWAPCHAIN_EXTENSION_NAME)
                 .set_minimum_version(1, 3);

  VkPhysicalDeviceVulkan13Features features13 = {};
  features13.sType                            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
  features13.dynamicRendering                 = true;
  features13.synchronization2                 = true;

  VkPhysicalDeviceVulkan12Features features12          = {};
  features12.bufferDeviceAddress                       = true;
  features12.runtimeDescriptorArray                    = true;
  features12.shaderSampledImageArrayNonUniformIndexing = true;

  VkPhysicalDeviceFeatures features = {};
  features.shaderInt64              = true;
  features.geometryShader           = true;
  features.samplerAnisotropy        = true;

  if (config.options.raytracing_enabled) {

    VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_feature = {};
    accel_feature.sType                                            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    accel_feature.accelerationStructure                            = true;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_feature = {};
    rt_pipeline_feature.sType                                         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rt_pipeline_feature.rayTracingPipeline                            = true;

    selector = selector //
                   .add_required_extension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)
                   .add_required_extension_features(rt_pipeline_feature)
                   .add_required_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
                   .add_required_extension_features(accel_feature)
                   .add_required_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
  }

  auto ph_device_result = selector //
                              .set_required_features_13(features13)
                              .set_required_features_12(features12)
                              .set_required_features(features)
                              .select();

  if (!ph_device_result.has_value()) {
    WERROR("Failed to select suitable physical device, message: {}", ph_device_result.error().message());
    throw std::runtime_error("Failed to select physical device");
  }

  m_device.physical = ph_device_result->physical_device;
  WINFO("physical device selected: {}", ph_device_result->properties.deviceName);

  vkb::DeviceBuilder device_builder{ ph_device_result.value() };
  auto               device_result = device_builder.build();

  if (!device_result.has_value()) {
    WERROR("Failed to create Vulkan Logical Device, message: {}", device_result.error().message());
    throw std::runtime_error("Failed to create VkDevice");
  }

  m_device.logical = device_result->device;
  WINFO("created logical device {}", (void*) m_device.logical);
  load_vk_extensions(m_instance, vkGetInstanceProcAddr, m_device.logical, vkGetDeviceProcAddr);

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

  VmaAllocatorCreateInfo alloc_create_info = {};
  alloc_create_info.vulkanApiVersion       = VK_API_VERSION_1_2;
  alloc_create_info.instance               = m_instance;
  alloc_create_info.physicalDevice         = m_device.physical;
  alloc_create_info.device                 = m_device.logical;
  alloc_create_info.flags                  = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

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
      "creating main command pool"
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
  m_swapchain.present_mode = swapchain_result->present_mode;
  m_swapchain.image_format = swapchain_result->image_format;
  m_swapchain.extent       = swapchain_result->extent;
  m_swapchain.image_count  = swapchain_result->image_count;

  auto swapchain_images      = swapchain_result->get_images().value();
  auto swapchain_image_views = swapchain_result->get_image_views().value();

  VkImageCreateInfo depth_img_info = {};
  depth_img_info.sType             = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  depth_img_info.imageType         = VK_IMAGE_TYPE_2D;
  depth_img_info.format            = m_swapchain.depth_format;
  depth_img_info.extent.width      = m_swapchain.extent.width;
  depth_img_info.extent.height     = m_swapchain.extent.height;
  depth_img_info.extent.depth      = 1;
  depth_img_info.mipLevels         = 1;
  depth_img_info.arrayLayers       = 1;
  depth_img_info.samples           = VK_SAMPLE_COUNT_1_BIT;
  depth_img_info.tiling            = VK_IMAGE_TILING_OPTIMAL;
  depth_img_info.usage             = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

  VmaAllocationCreateInfo vma_allocation_info = {};
  // FIXME: remove auto
  vma_allocation_info.usage = VMA_MEMORY_USAGE_AUTO;

  VkImageViewCreateInfo depth_view_info           = {};
  depth_view_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  depth_view_info.format                          = m_swapchain.depth_format;
  depth_view_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
  depth_view_info.subresourceRange.baseMipLevel   = 0;
  depth_view_info.subresourceRange.levelCount     = 1;
  depth_view_info.subresourceRange.baseArrayLayer = 0;
  depth_view_info.subresourceRange.layerCount     = 1;
  depth_view_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;

  m_frames.reserve(m_swapchain.image_count);
  for (u32 i = 0; i < m_swapchain.image_count; i += 1) {
    swapchain_frame_t frame = {};
    frame.image             = swapchain_images[i];
    frame.image_view        = swapchain_image_views[i];

    check(
        vmaCreateImage(
            m_vma,                                       //
            &depth_img_info, &vma_allocation_info,       //
            &frame.depth.image, &frame.depth.allocation, //
            nullptr
        ),
        fmt::format("creating depth image#{}", i)
    );
    depth_view_info.image = frame.depth.image;

    check(
        vkCreateImageView(m_device.logical, &depth_view_info, nullptr, &frame.depth.image_view), //
        fmt::format("creating image view for depth buffer#{}", i)
    );

    m_frames.push_back(frame);
  }

  set_debug_name(m_command_pool, "main command_pool");
  for (u32 i = 0; i < m_swapchain.image_count; i += 1) {
    set_debug_name(m_frames[i].image, fmt::format("swapchain_image #{}", i + 1));
    set_debug_name(m_frames[i].image_view, fmt::format("swapchain_image_view #{}", i + 1));
    set_debug_name(m_frames[i].depth.image, fmt::format("swapchain_depth_image #{}", i + 1));
    set_debug_name(m_frames[i].depth.image_view, fmt::format("swapchain_depth_image_view #{}", i + 1));
  }

  VkCommandPoolCreateInfo imm_cmd_pool_info = {};
  imm_cmd_pool_info.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  imm_cmd_pool_info.pNext                   = nullptr;
  imm_cmd_pool_info.queueFamilyIndex        = m_device.graphics_family_index;
  imm_cmd_pool_info.flags                   = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  check(
      vkCreateCommandPool(m_device.logical, &imm_cmd_pool_info, nullptr, &m_immediate_data.cmd_pool), //
      "creating command pool for immediate submission"
  );
  set_debug_name(m_immediate_data.cmd_pool, "immediate command pool");

  VkCommandBufferAllocateInfo cmd_buffers_create_info = {};
  cmd_buffers_create_info.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmd_buffers_create_info.commandPool                 = m_immediate_data.cmd_pool;
  cmd_buffers_create_info.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmd_buffers_create_info.commandBufferCount          = 1;

  check(
      vkAllocateCommandBuffers(
          m_device.logical, &cmd_buffers_create_info,
          &m_immediate_data.cmd_buffer
      ), //
      "allocating command buffer for immediate command pool"
  );
  set_debug_name(m_immediate_data.cmd_buffer, "immediate command buffer");

  VkFenceCreateInfo fence_create_info = {};
  fence_create_info.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_create_info.pNext             = nullptr;
  fence_create_info.flags             = VK_FENCE_CREATE_SIGNALED_BIT;

  check(
      vkCreateFence(m_device.logical, &fence_create_info, nullptr, &m_immediate_data.fence), //
      "creating fence for immediate cmd buffers"
  );
  set_debug_name(m_immediate_data.fence, "immediate fence");
}

Context::~Context() {
  // check if context is valid in case if it is moved
  if (m_instance != nullptr) {
    WINFO("Destroying vulkan context");
    /*
      ORDER OF DESTRUCTION:
        1. waiting until device is done touching our images
        2. destroying immediate data
        3. destroying of swapchain's image_views
        4. destroying of swapchain
        5. destroying of command_pool
        6. destroying of device
        7. destroying of debug message utils
        8. destroying of instance
    */

    vkDeviceWaitIdle(m_device.logical);

    vkDestroyFence(m_device.logical, m_immediate_data.fence, nullptr);
    vkFreeCommandBuffers(m_device.logical, m_immediate_data.cmd_pool, 1, &m_immediate_data.cmd_buffer);
    vkDestroyCommandPool(m_device.logical, m_immediate_data.cmd_pool, nullptr);

    for (auto const &frame : m_frames) {
      vkDestroyImageView(m_device.logical, frame.image_view, nullptr);
      vkDestroyImageView(m_device.logical, frame.depth.image_view, nullptr);
      vmaDestroyImage(m_vma, frame.depth.image, frame.depth.allocation);
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

// FIXME: && - explain yourself
void Context::immediate_submit(std::function<void(VkCommandBuffer cmd)> &&function) const {
  // FIXME: wtf am i doing here?
  check(vkResetFences(m_device.logical, 1, &m_immediate_data.fence), "reseting immediate fence");
  check(vkResetCommandBuffer(m_immediate_data.cmd_buffer, 0), "reseting immediate cmd buffers");

  VkCommandBuffer cmd = m_immediate_data.cmd_buffer;

  VkCommandBufferBeginInfo cmd_begin_info = {};
  cmd_begin_info.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  cmd_begin_info.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  check(vkBeginCommandBuffer(cmd, &cmd_begin_info), "beginning immediate command buffer");

  function(cmd);

  check(vkEndCommandBuffer(cmd), "ending immediate command buffer");

  VkCommandBufferSubmitInfo cmd_info = {};
  cmd_info.sType                     = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
  cmd_info.commandBuffer             = cmd;

  VkSubmitInfo2 submit          = {};
  submit.sType                  = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
  submit.commandBufferInfoCount = 1;
  submit.pCommandBufferInfos    = &cmd_info;

  // submit command buffer to the queue and execute it.
  //  fence will now block until the graphic commands finish execution
  check(vkQueueSubmit2(m_device.graphics_queue, 1, &submit, m_immediate_data.fence), "submiting immediate cmd to queue");

  check(vkWaitForFences(m_device.logical, 1, &m_immediate_data.fence, true, 9999999999), "waiting for fence");
}

// STD::SPAN SUCKS LITERALLY PIESE OF GARBAGE
image_t Context::create_image_on_gpu(VkImageCreateInfo image_info, u8* data, size_t size) {
  WASSERT(size != 0, "zero size not allowed");
  buffer_t staging = {};

  VkBufferCreateInfo staging_info = {};
  staging_info.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  staging_info.size               = size;
  staging_info.usage              = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  staging_info.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo staging_alloc{};
  staging_alloc.usage          = VMA_MEMORY_USAGE_CPU_TO_GPU;
  staging_alloc.flags          = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  staging_alloc.preferredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  check(
      vmaCreateBuffer(m_vma, &staging_info, &staging_alloc, &staging.handle, &staging.allocation, nullptr), //
      "creating staging buffer"
  );

  void* mapped_data = nullptr;
  vmaMapMemory(m_vma, staging.allocation, &mapped_data);
  memcpy(mapped_data, data, size);
  vmaUnmapMemory(m_vma, staging.allocation);

  image_t result = {};

  VmaAllocationCreateInfo result_alloc{};
  result_alloc.usage = VMA_MEMORY_USAGE_GPU_ONLY;

  check(
      vmaCreateImage(m_vma, &image_info, &result_alloc, &result.handle, &result.allocation, nullptr), //
      "creating result image"
  );

  immediate_submit([&](VkCommandBuffer cmd) {
    VkImageSubresourceRange subresource_range{};
    subresource_range.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    subresource_range.baseArrayLayer = 0;
    subresource_range.baseMipLevel   = 0;
    subresource_range.layerCount     = 1;
    subresource_range.levelCount     = image_info.mipLevels;

    transition_image(cmd, result.handle, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresource_range);

    VkImageSubresourceLayers subresource{};
    subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresource.layerCount = 1;

    VkBufferImageCopy copy = {};
    copy.imageSubresource  = subresource;
    copy.imageExtent       = image_info.extent;

    vkCmdCopyBufferToImage(cmd, staging.handle, result.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
  });

  vmaDestroyBuffer(m_vma, staging.handle, staging.allocation);
  return result;
}

void Context::generate_mipmaps(VkImage image, VkImageCreateInfo image_info) {
  WASSERT(image != VK_NULL_HANDLE, "invalid image handle");

  immediate_submit([&](VkCommandBuffer cmd) {
    VkImageMemoryBarrier barrier{};
    barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel   = 0;
    barrier.subresourceRange.layerCount     = 1;
    barrier.subresourceRange.levelCount     = 1;
    barrier.image                           = image;
    barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;

    i32 mip_width  = image_info.extent.width;
    i32 mip_height = image_info.extent.height;
    for (u32 i = 1; i < image_info.mipLevels; i += 1) {
      barrier.subresourceRange.baseMipLevel = i - 1;
      barrier.oldLayout                     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      barrier.newLayout                     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      barrier.srcAccessMask                 = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask                 = VK_ACCESS_TRANSFER_READ_BIT;
      vkCmdPipelineBarrier(
          cmd,                            // cmd
          VK_PIPELINE_STAGE_TRANSFER_BIT, // source stage
          VK_PIPELINE_STAGE_TRANSFER_BIT, // destination stage
          0,                              // dependencyFlags
          0,                              // memoryBarrierCount
          nullptr,                        // pMemoryBarriers
          0,                              // bufferMemoryBarrierCount
          nullptr,                        // pBufferMemoryBarriers
          1,                              // imageMemoryBarrierCount
          &barrier                        //  pImageMemoryBarriers
      );

      VkImageBlit blit{};
      blit.srcOffsets[0]                 = { 0, 0, 0 };
      blit.srcOffsets[1]                 = { mip_width, mip_height, 1 };
      blit.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
      blit.srcSubresource.mipLevel       = i - 1;
      blit.srcSubresource.baseArrayLayer = 0;
      blit.srcSubresource.layerCount     = 1;
      blit.dstOffsets[0]                 = { 0, 0, 0 };
      blit.dstOffsets[1]                 = { mip_width > 1 ? mip_width / 2 : 1, mip_height > 1 ? mip_height / 2 : 1, 1 };
      blit.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
      blit.dstSubresource.mipLevel       = i;
      blit.dstSubresource.baseArrayLayer = 0;
      blit.dstSubresource.layerCount     = 1;

      vkCmdBlitImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);

      barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

      if (mip_width > 1) mip_width /= 2;
      if (mip_height > 1) mip_height /= 2;
    }

    barrier.subresourceRange.baseMipLevel = image_info.mipLevels - 1;
    barrier.oldLayout                     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout                     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask                 = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask                 = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
  });
}

VkDeviceAddress Context::get_buffer_device_address(VkBuffer buffer) const {
  WASSERT(buffer != VK_NULL_HANDLE, "buffer should be valid");

  VkBufferDeviceAddressInfo info = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
  info.buffer                    = buffer;
  return vkGetBufferDeviceAddress(m_device.logical, &info);
}

buffer_t Context::create_buffer(
    VkDeviceSize size, const void* data, //
    VkBufferUsageFlags usage, VkMemoryPropertyFlags mem_props
) {
  WASSERT(size != 0, "zero size not allowed");
  buffer_t staging = {};

  VkBufferCreateInfo staging_buffer_info = {};
  staging_buffer_info.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  staging_buffer_info.size               = size;
  staging_buffer_info.usage              = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  staging_buffer_info.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo staging_buffer_alloc = {};
  staging_buffer_alloc.usage                   = VMA_MEMORY_USAGE_CPU_TO_GPU;
  staging_buffer_alloc.flags                   = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  staging_buffer_alloc.preferredFlags          = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  check(
      vmaCreateBuffer(
          m_vma,                                       //
          &staging_buffer_info, &staging_buffer_alloc, //
          &staging.handle, &staging.allocation, nullptr
      ),
      "creating staging buffer"
  );

  void* mapped_data = nullptr;
  vmaMapMemory(m_vma, staging.allocation, &mapped_data);
  memcpy(mapped_data, data, size);
  vmaUnmapMemory(m_vma, staging.allocation);

  buffer_t result = {};

  VkBufferCreateInfo result_buffer_info = {};
  result_buffer_info.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  result_buffer_info.size               = size;
  result_buffer_info.usage              = VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage;
  result_buffer_info.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo result_buffer_alloc = {};
  result_buffer_alloc.usage                   = VMA_MEMORY_USAGE_GPU_ONLY;
  result_buffer_alloc.requiredFlags           = mem_props;

  check(
      vmaCreateBuffer(
          m_vma,                                     //
          &result_buffer_info, &result_buffer_alloc, //
          &result.handle, &result.allocation, nullptr
      ),
      "creating destination buffer for transferring"
  );

  immediate_submit([&](VkCommandBuffer cmd) {
    VkBufferCopy copy_region{};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size      = size;

    vkCmdCopyBuffer(cmd, staging.handle, result.handle, 1, &copy_region);
  });

  vmaDestroyBuffer(m_vma, staging.handle, staging.allocation);

  return result;
}

VkShaderModule Context::create_shader_module(std::string_view file_path) const {

  if (!std::filesystem::exists(file_path)) {
    WERROR("Failed to create ShaderModule: file not found {}", file_path);
    throw std::runtime_error("failed to create shader module");
  }

  std::ifstream file{ file_path.data(), std::ios::binary | std::ios::ate };

  if (!file.is_open()) {
    WERROR("Failed to create ShaderModule: cant open file {}", file_path);
    throw std::runtime_error("failed to create shader module");
  }

  std::size_t       file_size = file.tellg();
  std::vector<char> buffer(file_size);

  file.seekg(0);
  file.read(buffer.data(), file_size);
  file.close();

  VkShaderModuleCreateInfo create_info = {};

  create_info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = buffer.size();
  create_info.pCode    = (u32*) buffer.data();

  VkShaderModule shader_module = nullptr;
  check(
      vkCreateShaderModule(m_device.logical, &create_info, nullptr, &shader_module), //
      fmt::format("creating shader module from file {}", file_path)
  );

  return shader_module;
}

void Context::transition_image(
    VkCommandBuffer cmd, VkImage image,                   //
    VkImageLayout currentLayout, VkImageLayout newLayout, //
    VkImageSubresourceRange subresource
) const {
  WASSERT(cmd != VK_NULL_HANDLE, "invalid command buffer");
  WASSERT(image != VK_NULL_HANDLE, "invalid image handke");
  VkImageMemoryBarrier2 image_barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
  image_barrier.pNext = nullptr;

  image_barrier.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  image_barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
  image_barrier.dstStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  image_barrier.dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;

  image_barrier.oldLayout = currentLayout;
  image_barrier.newLayout = newLayout;

  image_barrier.subresourceRange = subresource;
  image_barrier.image            = image;

  VkDependencyInfo dep_info{};
  dep_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
  dep_info.pNext = nullptr;

  dep_info.imageMemoryBarrierCount = 1;
  dep_info.pImageMemoryBarriers    = &image_barrier;

  vkCmdPipelineBarrier2(cmd, &dep_info);
}

void Context::transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout) const {

  WASSERT(cmd != VK_NULL_HANDLE, "invalid command buffer");
  WASSERT(image != VK_NULL_HANDLE, "invalid image handke");

  VkImageAspectFlags aspect_mask = (newLayout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;

  VkImageSubresourceRange subresource{};
  subresource.aspectMask     = aspect_mask;
  subresource.baseMipLevel   = 0;
  subresource.levelCount     = VK_REMAINING_MIP_LEVELS;
  subresource.baseArrayLayer = 0;
  subresource.layerCount     = VK_REMAINING_ARRAY_LAYERS;

  transition_image(cmd, image, currentLayout, newLayout, subresource);
}

void Context::set_debug_name(VkImage image, std::string_view name) const {

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

void Context::set_debug_name(VkImageView image_view, std::string_view name) const {

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

void Context::set_debug_name(VkCommandPool command_pool, std::string_view name) const {

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

void Context::set_debug_name(VkCommandBuffer command_buffer, std::string_view name) const {

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

void Context::set_debug_name(VkFramebuffer frame_buffer, std::string_view name) const {

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

void Context::set_debug_name(VkFence fence, std::string_view name) const {

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

void Context::set_debug_name(VkSemaphore semaphore, std::string_view name) const {

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

void Context::set_debug_name(VkPipeline pipeline, std::string_view name) const {

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

void Context::set_debug_name(VkBuffer buffer, std::string_view name) const {

  VkDebugUtilsObjectNameInfoEXT info = {};
  info.sType                         = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  info.objectType                    = VK_OBJECT_TYPE_BUFFER;
  info.objectHandle                  = (uint64_t) buffer;
  info.pObjectName                   = name.data();

  check(
      vkSetDebugUtilsObjectNameEXT(m_device.logical, &info), //
      fmt::format("setting name:{} to VkBuffer:{}", name, fmt::ptr(buffer))
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

[[nodiscard]] std::vector<swapchain_frame_t> const &Context::swapchain_frames() const { return m_frames; }

[[nodiscard]] VkPresentModeKHR Context::swapchain_present_mode() const { return m_swapchain.present_mode; }

[[nodiscard]] VkFormat Context::swapchain_image_format() const { return m_swapchain.image_format; }

[[nodiscard]] VkFormat Context::swapchain_depth_format() const { return m_swapchain.depth_format; }

[[nodiscard]] VkExtent2D Context::swapchain_extent() const { return m_swapchain.extent; }

[[nodiscard]] u32 Context::swapchain_image_count() const { return m_swapchain.image_count; }

[[nodiscard]] GLFWwindow* Context::window() const { return m_window_ref.get().handle(); }

} // namespace whim::vk
