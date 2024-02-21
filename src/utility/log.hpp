
#include <cstdlib>
#include "fmt/format.h"

namespace whim {

#define WERROR(message, ...) fmt::println(stderr, "[ERROR] " message, __VA_ARGS__) // NOLINT
#define WINFO(message, ...)  fmt::println("[INFO]  " message, __VA_ARGS__)         // NOLINT
#define WASSERT(exp, msg)                                                                                                 \
  do {                                                                                                                    \
    if (!(exp)) {                                                                                                         \
      fmt::println(stderr, "[ASSERT] {} \n    Expected: {} \n    Source: {}, line: {}", (msg), #exp, __FILE__, __LINE__); \
      std::abort();                                                                                                       \
    }                                                                                                                     \
  } while (false)

} // namespace whim
