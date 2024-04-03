#pragma once

#include <cstdint>
#include <memory>

namespace whim {

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using f32 = float;
using f64 = double;

using usize = std::size_t;

template<typename T>
using sptr = std::shared_ptr<T>;

template<typename T>
using uptr = std::unique_ptr<T>;

template<typename T>
concept NonConst = not std::is_const_v<T>;

template<typename T>
concept NonReference = not std::is_reference_v<T>;

/*
  This is used because references in classes break move semantics
*/
template<typename T>
  requires NonReference<T>
using ref = std::reference_wrapper<T>;

template<typename T>
  requires NonConst<T> and NonReference<T>
using cref = std::reference_wrapper<T const>;

/*
  This class used for pointer like handlers (GLFWwindow*, VkImage, etc...)
  MoveHandle automatically invalidate this handlers on move

  on move: old_handle == invalid_state
*/
template<typename T, T invalid_state>
class MoveHandle {

public:
  constexpr MoveHandle()  = default;
  constexpr ~MoveHandle() = default;

  constexpr MoveHandle(T handle) : // NOLINT its okay to be implicit
      m_handle(handle) {}

  constexpr MoveHandle(MoveHandle const &other)            = default;
  constexpr MoveHandle &operator=(MoveHandle const &other) = default;

  constexpr MoveHandle(MoveHandle &&other) noexcept :
      m_handle(std::exchange(other.m_handle, invalid_state)) {}

  constexpr MoveHandle &operator=(MoveHandle &&other) noexcept {
    // since operator& is overloaded, we cant use (this != &other) condition
    if (this->m_handle != other.m_handle) {
      m_handle = std::exchange(other.m_handle, invalid_state);
    }
    return *this;
  }

  constexpr MoveHandle &operator=(T value) {
    m_handle = value;
    return *this;
  }

  constexpr operator T() const { return m_handle; }                  // NOLINT its okay to be implicit

  constexpr operator bool() const { return m_handle != invalid_state; } // NOLINT its okay to be implicit

  T const* operator&() const { return &m_handle; }                   // NOLINT i know what im doing xdd

  T* operator&() { return &m_handle; }                               // NOLINT i know what im doing xdd

private:
  T m_handle = invalid_state;
};

template<typename T>
using ptr = MoveHandle<T*, nullptr>;
} // namespace whim
