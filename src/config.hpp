

#pragma once

#include "utility/macros.hpp"
#include "utility/types.hpp"
#include <string>

struct config_t {

  whim::u32   width    = 1;
  whim::u32   height   = 1;
  std::string app_name = "default_name";

  struct options_t {
    bool is_resizable              = false;
    bool is_fullscreen             = false;
    bool validation_layers_support = true;
    bool raytracing_enabled        = true;

  } options;
};

inline void assert_config(config_t const &config) { UNUSED(config); }
