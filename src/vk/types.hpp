#pragma once

#include <vulkan/vulkan_core.h>
#include <vma/vk_mem_alloc.h>

#include "whim.hpp"

namespace whim::vk {

template<typename VulkanType>
using handle = MoveHandle<VulkanType, VK_NULL_HANDLE>;

struct buffer_t {
  handle<VkBuffer>                  handle     = VK_NULL_HANDLE;
  ::whim::vk::handle<VmaAllocation> allocation = VK_NULL_HANDLE;
};

struct image_t {
  handle<VkImage>                   handle     = VK_NULL_HANDLE;
  ::whim::vk::handle<VmaAllocation> allocation = VK_NULL_HANDLE;
};

struct acceleration_structure_t {
  handle<VkAccelerationStructureKHR> handle = VK_NULL_HANDLE;
  buffer_t                           buffer = {};
};

struct texture_t {
    image_t             image   = {};
    handle<VkImageView> view    = VK_NULL_HANDLE;
    handle<VkSampler>   sampler = VK_NULL_HANDLE;

    VkFormat format = VK_FORMAT_R8G8B8A8_SRGB;
    u32      width  = 0;
    u32      height = 0;
  };

} // namespace whim::vk