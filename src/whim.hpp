#pragma once

#include "utility/log.hpp"
#include "utility/types.hpp"
#include "utility/macros.hpp"
#include "utility/align.hpp"

#include "vk/types.hpp"
#include "vk/result.hpp"

// TODO: move this to right place
namespace whim {
inline uint32_t mip_levels(u32 width, u32 height) { //
  return (u32) (std::floor(std::log2(std::max(width, height)))) + 1;
}
} // namespace whim
