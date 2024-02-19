#pragma once

#include <stdexcept>
#include <string_view>
#include <Volk/volk.h>

#include "utility/log.hpp"

namespace whim::vk {

inline void check(VkResult result, std::string_view description) {
  if (result != VK_SUCCESS) {
    WERROR("[ERROR] Operation failed, description: {}", description);
    throw std::runtime_error("Some Vulkan function failed");
  }
}

} // namespace whim::vk
