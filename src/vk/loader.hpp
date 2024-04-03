#pragma once
#include <vulkan/vulkan.h>

namespace whim::vk {

void load_vk_extensions(
    VkInstance instance, PFN_vkGetInstanceProcAddr getInstanceProcAddr, //
    VkDevice device, PFN_vkGetDeviceProcAddr getDeviceProcAddr
);
}