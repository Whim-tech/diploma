#pragma once

#include <vulkan/vulkan_core.h>
#include <vma/vk_mem_alloc.h>

#include "whim.hpp"

namespace whim::vk {

template<typename VulkanType>
using handle = MoveHandle<VulkanType, VK_NULL_HANDLE>;

struct buffer_t {
  handle<VkBuffer>                  handle     = nullptr;
  // wtf is going on?...
  ::whim::vk::handle<VmaAllocation> allocation = nullptr;
};

struct acceleration_structure {
  handle<VkAccelerationStructureKHR> handle = VK_NULL_HANDLE;
  buffer_t                           buffer = {};
};

} // namespace whim::vk