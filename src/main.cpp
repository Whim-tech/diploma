#include <cassert>
#include <iostream>

#define VOLK_IMPLEMENTATION
#include <Volk/volk.h>

#include <VkBootstrap.h>

#include <GLFW/glfw3.h>

int main() {
  std::cout << "hello, world!\n";

  volkInitialize();
  glfwInit();
  assert(glfwVulkanSupported());
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

  GLFWwindow* window = glfwCreateWindow(600, 480, "test volk", nullptr, nullptr);

  uint32_t                 extensions_count = 0;
  const char**             glfw_extensions  = glfwGetRequiredInstanceExtensions(&extensions_count);
  std::vector<const char*> extensions(glfw_extensions, glfw_extensions + extensions_count); // NOLINT

  vkb::InstanceBuilder builder;

  auto inst_ret = builder                                         //
                      .set_app_name("Example Vulkan Application") //
                      .require_api_version(1, 3)                  //
                      .request_validation_layers()                //
                      .enable_extensions(extensions)
                      .use_default_debug_messenger()
                      .build();

  vkb::Instance vkb_inst = inst_ret.value();
  volkLoadInstance(vkb_inst.instance);

  VkSurfaceKHR surface = VK_NULL_HANDLE;
  glfwCreateWindowSurface(vkb_inst.instance, window, nullptr, &surface);

  vkb::PhysicalDeviceSelector                      selector{ vkb_inst };
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_feature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR    rt_pipeline_feature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };
  auto                                             phys_ret = selector //
                      .set_surface(surface)
                      .set_minimum_version(1, 3)
                      .require_dedicated_transfer_queue()
                      .add_required_extension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)
                      .add_required_extension_features(rt_pipeline_feature)
                      .add_required_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
                      .add_required_extension_features(accel_feature)
                      .add_required_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)
                      .select();

  vkb::DeviceBuilder device_builder{ phys_ret.value() };
  auto               dev_ret    = device_builder.build();
  vkb::Device        vkb_device = dev_ret.value();

  VkDevice device = vkb_device.device;

  volkLoadDevice(device);

  // Get the graphics queue with a helper function
  auto    graphics_queue_ret = vkb_device.get_queue(vkb::QueueType::graphics);
  VkQueue graphics_queue     = graphics_queue_ret.value();

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR };
  // Requesting ray tracing properties
  VkPhysicalDeviceProperties2 prop2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
  prop2.pNext = &m_rtProperties;
  vkGetPhysicalDeviceProperties2(vkb_device.physical_device, &prop2);

  vkDestroyDebugUtilsMessengerEXT(vkb_inst.instance, vkb_inst.debug_messenger, nullptr);

  return 0;
}