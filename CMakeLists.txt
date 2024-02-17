cmake_minimum_required(VERSION 3.28)

project(raytracing CXX)

option(WHIM_BUILD_TEST "Build tests" ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
  set(WHIM_DEFAULT_COMPILE_OPTIONS /W4)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(WHIM_DEFAULT_COMPILE_OPTIONS -Wall -Wextra -Wpedantic -glldb -Wunknown-warning-option)
else()
  set(WHIM_DEFAULT_COMPILE_OPTIONS -Wall -Wextra -Wpedantic)
endif()

# clangd cant recognize /std:c++latest or -std:c++latest
# so i can use only c++20
# https://github.com/clangd/clangd/issues/1383
# FIXME: find some way to make clang dont ignore my std flags with msvc
set(WHIM_DEFAULT_COMPILE_FEATURE "cxx_std_20")
set(WHIM_EXTERNAL_DEPS_PATH "${PROJECT_SOURCE_DIR}/external")
set(BASE_DIRECTORY "${PROJECT_SOURCE_DIR}/external" )

add_subdirectory("${PROJECT_SOURCE_DIR}/external/volk")
add_subdirectory("${PROJECT_SOURCE_DIR}/external/vk-bootstrap")
add_subdirectory("${PROJECT_SOURCE_DIR}/external/VulkanMemoryAllocator")

add_subdirectory("${PROJECT_SOURCE_DIR}/external/fmt")

# GLFW
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("${PROJECT_SOURCE_DIR}/external/glfw")
# GLM
add_subdirectory("${PROJECT_SOURCE_DIR}/external/glm")

# this should be added after glfw and vulkan
# DO NOT CHANGE ORDER OF THIS WITH VULKAN OR GLFW!
# add_subdirectory("${PROJECT_SOURCE_DIR}/external/imgui")


add_executable(main)
target_compile_options(main PRIVATE ${WHIM_DEFAULT_COMPILE_OPTIONS})
target_compile_features(main PRIVATE ${WHIM_DEFAULT_COMPILE_FEATURE})
target_include_directories(main PRIVATE "${PROJECT_SOURCE_DIR}/src")


file(GLOB_RECURSE source_files CONFIGURE_DEPENDS "src/*.cpp" "src/*.cxx" "src/*.cc")
target_sources(main PRIVATE ${source_files})

target_link_libraries(main PRIVATE
  glfw
  vk-bootstrap::vk-bootstrap
  glm::glm
  fmt::fmt
  GPUOpen::VulkanMemoryAllocator # vma
#   imgui
)

set_target_properties(main PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/bin"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/bin"
)