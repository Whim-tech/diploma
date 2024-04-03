#include "vk/raytracer.hpp"
#include "fmt/format.h"
#include "shader.h"
#include <glm/gtc/type_ptr.hpp>

#include <array>
#include <filesystem>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_vulkan.h"
#include "imgui/imgui_impl_glfw.h"

#include <tiny_obj_loader.h>

namespace whim::vk {

RayTracer::RayTracer(Context &context, CameraManipulator const &man) :
    m_context_ref(context),
    m_camera_ref(man) {

  create_frame_data();
  init_imgui();
}

void RayTracer::draw() {
  Context const &context = m_context_ref;

  // ---------- IMGUI ----------------
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  ImGui::ShowDemoWindow();

  ImGui::Render();

  // --------- GETTING AN IMAGE -----------------
  constexpr u64       no_timeout = std::numeric_limits<u64>::max();
  render_frame_data_t frame      = m_frames[m_current_frame];
  // wait until the gpu has finished rendering the last frame
  check(
      vkWaitForFences(context.device(), 1, &frame.fence, true, no_timeout), //
      fmt::format("waiting for render fence #{}", m_current_frame)
  );

  u32 image_index = 0;
  check(
      vkAcquireNextImageKHR(
          context.device(),      //
          context.swapchain(),   //
          no_timeout,            //
          frame.image_semaphore, //
          nullptr,               //
          &image_index
      ),                         //
      "acquiring next image index from swapchain"
  );

  // -------- BEFORE FRAME ------------------
  check(vkResetCommandBuffer(frame.cmd, 0), "");
  VkCommandBufferBeginInfo begin_info = {};
  begin_info.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags                    = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  begin_info.pInheritanceInfo         = nullptr;

  check(
      vkBeginCommandBuffer(frame.cmd, &begin_info), //
      fmt::format("beginning rendering frame#{}", m_current_frame)
  );

  context.transition_image(frame.cmd, context.swapchain_frames()[image_index].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
  context.transition_image(frame.cmd, context.swapchain_frames()[image_index].depth.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

  // -------- RENDERING ---------------------
  VkRect2D render_area = {
    .offset = VkOffset2D{0, 0},
      .extent = context.swapchain_extent()
  };

  VkRenderingAttachmentInfo color_attachment = {};
  color_attachment.sType                     = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
  color_attachment.imageView                 = context.swapchain_frames()[image_index].image_view;
  color_attachment.imageLayout               = VK_IMAGE_LAYOUT_GENERAL;
  color_attachment.loadOp                    = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color_attachment.storeOp                   = VK_ATTACHMENT_STORE_OP_STORE;
  color_attachment.clearValue.color          = {
    {0.0f, 0.0f, 0.0f, 1.0f}
  };

  VkRenderingAttachmentInfo depth_attachment     = {};
  depth_attachment.sType                         = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
  depth_attachment.imageView                     = context.swapchain_frames()[image_index].depth.image_view;
  depth_attachment.imageLayout                   = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
  depth_attachment.loadOp                        = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depth_attachment.storeOp                       = VK_ATTACHMENT_STORE_OP_STORE;
  depth_attachment.clearValue.depthStencil.depth = 1.f;

  VkRenderingInfo render_info      = {};
  render_info.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
  render_info.layerCount           = 1;
  render_info.colorAttachmentCount = 1;
  render_info.pColorAttachments    = &color_attachment;
  render_info.pDepthAttachment     = &depth_attachment;
  render_info.pStencilAttachment   = nullptr;
  render_info.renderArea           = render_area;

  vkCmdBeginRendering(frame.cmd, &render_info);
  // ------------ DRAWING IN THERE -----------------
  vkCmdEndRendering(frame.cmd);

  // --------------- IMGUI RENDERING-------------
  VkRenderingAttachmentInfo imgui_color_attachment = {};
  imgui_color_attachment.sType                     = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
  imgui_color_attachment.imageView                 = context.swapchain_frames()[image_index].image_view;
  imgui_color_attachment.imageLayout               = VK_IMAGE_LAYOUT_GENERAL;
  imgui_color_attachment.loadOp                    = VK_ATTACHMENT_LOAD_OP_LOAD;
  imgui_color_attachment.storeOp                   = VK_ATTACHMENT_STORE_OP_STORE;

  VkRenderingInfo imgui_render_info      = {};
  imgui_render_info.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
  imgui_render_info.layerCount           = 1;
  imgui_render_info.colorAttachmentCount = 1;
  imgui_render_info.renderArea           = render_area;
  imgui_render_info.pColorAttachments    = &imgui_color_attachment;
  imgui_render_info.pDepthAttachment     = nullptr;
  imgui_render_info.pStencilAttachment   = nullptr;

  vkCmdBeginRendering(frame.cmd, &imgui_render_info);
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), frame.cmd);
  vkCmdEndRendering(frame.cmd);

  // ------------- AFTER FRAME ----------------
  context.transition_image(frame.cmd, context.swapchain_frames()[image_index].image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

  check(vkEndCommandBuffer(frame.cmd), fmt::format("ending rendering frame#{}", m_current_frame));

  // ---------- SUBMITTING -----------------
  VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

  VkSubmitInfo submit_info         = {};
  submit_info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.waitSemaphoreCount   = 1;
  submit_info.pWaitSemaphores      = &frame.image_semaphore;
  submit_info.pWaitDstStageMask    = &wait_stage;
  submit_info.commandBufferCount   = 1;
  submit_info.pCommandBuffers      = &frame.cmd;
  submit_info.signalSemaphoreCount = 1;
  submit_info.pSignalSemaphores    = &frame.render_semaphore;

  check(vkResetFences(context.device(), 1, &frame.fence), "reseting fence");

  check(
      vkQueueSubmit(
          context.graphics_queue(), //
          1, &submit_info,          //
          frame.fence               //
      ),
      fmt::format("submitting {} image to graphics queue on frame{}", image_index, m_current_frame)
  );

  VkSwapchainKHR swapchain = context.swapchain();

  VkPresentInfoKHR present_info   = {};
  present_info.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores    = &frame.render_semaphore;
  present_info.swapchainCount     = 1;
  present_info.pSwapchains        = &swapchain;
  present_info.pImageIndices      = &image_index;
  present_info.pResults           = nullptr; // Optional

  check(
      vkQueuePresentKHR(context.present_queue(), &present_info), //
      fmt::format("submitting {} image to present queue in frame{}", image_index, m_current_frame)
  );

  m_current_frame = (m_current_frame + 1) / max_frames;
}

RayTracer::~RayTracer() {
  // TODO: check if raytracer is still valid (after move)
  if (true) {
    /*
      ORDER OF CLEAN UP:

      1 - wait for device
      2 - description buffer destruction
      2 - BLAS cleanup
      3 - imgui cleanup
      4 - frame data cleanup
    */
    Context const &context = m_context_ref;

    vkDeviceWaitIdle(context.device());

    vmaDestroyBuffer(context.vma_allocator(), m_description.buffer.handle, m_description.buffer.allocation);

    for (auto const &[k, v] : m_meshes) {
      vmaDestroyBuffer(context.vma_allocator(), v.blas.buffer.handle, v.blas.buffer.allocation);
      vkDestroyAccelerationStructureKHR(context.device(), v.blas.handle, nullptr);

      vmaDestroyBuffer(context.vma_allocator(), v.gpu.index.handle, v.gpu.index.allocation);
      vmaDestroyBuffer(context.vma_allocator(), v.gpu.vertex.handle, v.gpu.vertex.allocation);
      vmaDestroyBuffer(context.vma_allocator(), v.gpu.material.handle, v.gpu.material.allocation);
      vmaDestroyBuffer(context.vma_allocator(), v.gpu.material_index.handle, v.gpu.material_index.allocation);
    }

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    vkDestroyDescriptorPool(context.device(), m_imgui.desc_pool, nullptr);

    std::vector<VkCommandBuffer> buffers{ (size_t) max_frames };
    for (auto frame_data : m_frames) {
      buffers.push_back(frame_data.cmd);
      vkDestroyFence(context.device(), frame_data.fence, nullptr);
      vkDestroySemaphore(context.device(), frame_data.image_semaphore, nullptr);
      vkDestroySemaphore(context.device(), frame_data.render_semaphore, nullptr);
    }

    vkFreeCommandBuffers(
        context.device(),       //
        context.command_pool(), //
        max_frames,             //
        buffers.data()
    );
  }
}

void RayTracer::create_frame_data() {
  Context const &context = m_context_ref.get();

  VkCommandBufferAllocateInfo cmd_buffers_create_info = {};
  cmd_buffers_create_info.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmd_buffers_create_info.commandPool                 = context.command_pool();
  cmd_buffers_create_info.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmd_buffers_create_info.commandBufferCount          = max_frames;

  std::vector<VkCommandBuffer> buffers{ (size_t) max_frames };
  check(
      vkAllocateCommandBuffers(
          context.device(), &cmd_buffers_create_info,
          buffers.data()
      ), //
      "allocating render command buffers"
  );

  VkFenceCreateInfo fence_create_info = {};
  fence_create_info.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_create_info.pNext             = nullptr;
  fence_create_info.flags             = VK_FENCE_CREATE_SIGNALED_BIT;

  VkSemaphoreCreateInfo semaphore_create_info = {};
  semaphore_create_info.sType                 = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  semaphore_create_info.pNext                 = nullptr;
  semaphore_create_info.flags                 = 0;

  for (u32 i = 0; i < (u32) max_frames; i += 1) {
    render_frame_data_t data = {};
    check(
        vkCreateSemaphore(context.device(), &semaphore_create_info, nullptr, &data.image_semaphore), //
        fmt::format("creating image_semaphore #{}", i)
    );
    check(
        vkCreateSemaphore(context.device(), &semaphore_create_info, nullptr, &data.render_semaphore), //
        fmt::format("creating render_semaphore #{}", i)
    );
    check(
        vkCreateFence(context.device(), &fence_create_info, nullptr, &data.fence), //
        fmt::format("creating render fence#{}", i)
    );
    data.cmd = buffers[i];

    m_frames.push_back(data);

    context.set_debug_name(data.cmd, fmt::format("render cmd buffer #{}", i));
    context.set_debug_name(data.render_semaphore, fmt::format("render_semaphore #{}", i));
    context.set_debug_name(data.image_semaphore, fmt::format("image_semaphore #{}", i));
    context.set_debug_name(data.fence, fmt::format("in_flight_fence #{}", i));
  }
}

void RayTracer::init_imgui() {

  Context &context = m_context_ref.get();

  // IMGUI
  std::array<VkDescriptorPoolSize, 11> pool_sizes = {
    VkDescriptorPoolSize{               VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
    VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
    VkDescriptorPoolSize{         VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
    VkDescriptorPoolSize{         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
    VkDescriptorPoolSize{  VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
    VkDescriptorPoolSize{  VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
    VkDescriptorPoolSize{        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
    VkDescriptorPoolSize{        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
    VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
    VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
    VkDescriptorPoolSize{      VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}
  };

  VkDescriptorPoolCreateInfo imgui_pool_info = {};
  imgui_pool_info.sType                      = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  imgui_pool_info.flags                      = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  imgui_pool_info.maxSets                    = 1000;
  imgui_pool_info.poolSizeCount              = (uint32_t) std::size(pool_sizes);
  imgui_pool_info.pPoolSizes                 = pool_sizes.data();

  check(
      vkCreateDescriptorPool(context.device(), &imgui_pool_info, nullptr, &m_imgui.desc_pool), //
      "allocating descriptor pool for imgui"
  );

  ImGui_ImplVulkan_LoadFunctions(
      [](const char* function_name, void* p_context) {
        Context* context = (Context*) p_context;

        auto instance_addr = vkGetInstanceProcAddr(context->instance(), function_name);
        auto device_addr   = vkGetDeviceProcAddr(context->device(), function_name);

        return device_addr ? device_addr : instance_addr;
      },
      &context
  );

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void) io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;     // Enable Docking
  // ITS JUST DOESNT WORK =)
  // io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;   // Enable Multi-Viewport / Platform Windows

  ImGuiStyle &style = ImGui::GetStyle();
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
    style.WindowRounding              = 0.0f;
    style.Colors[ImGuiCol_WindowBg].w = 1.0f;
  }

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();

  ImGui_ImplGlfw_InitForVulkan(context.window(), true);

  // this initializes imgui for Vulkan
  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.Instance                  = context.instance();
  init_info.PhysicalDevice            = context.physical_device();
  init_info.Device                    = context.device();
  init_info.Queue                     = context.graphics_queue();
  init_info.DescriptorPool            = m_imgui.desc_pool;
  init_info.MinImageCount             = context.swapchain_image_count();
  init_info.ImageCount                = context.swapchain_image_count();
  init_info.UseDynamicRendering       = true;
  init_info.ColorAttachmentFormat     = context.swapchain_image_format();
  init_info.MSAASamples               = VK_SAMPLE_COUNT_1_BIT;

  ImGui_ImplVulkan_Init(&init_info, VK_NULL_HANDLE);

  ImGui_ImplVulkan_CreateFontsTexture();
}

void RayTracer::load_mesh(std::string_view file_path, std::string_view mesh_name) {
  Mesh mesh = {};

  WINFO("loading {} from file:{}", mesh_name, file_path);
  load_obj_file(file_path, mesh.raw);
  WINFO("loading {} to gpu", mesh_name);
  load_mesh_to_gpu(mesh);
  WINFO("creating blas for {}", mesh_name);
  load_mesh_to_blas(mesh);

  m_meshes[std::string{ mesh_name }] = mesh;
}

void RayTracer::load_obj_file(std::string_view file_path, Mesh::raw_data &mesh) {
  if (!std::filesystem::exists(file_path)) {
    WERROR("Cant parse obj file: file not found - {}", file_path);
    throw std::runtime_error("cant parse obj file");
  }

  tinyobj::ObjReader reader;
  reader.ParseFromFile(std::string(file_path));

  if (!reader.Valid()) {
    WERROR("Cant parse obj file: file is not valid: {}:{}", file_path, reader.Error());
    throw std::runtime_error("cant parse obj file");
  }

  // Collecting the material in the scene
  for (const auto &mat : reader.GetMaterials()) {
    material material      = {};
    material.ambient       = glm::vec3(mat.ambient[0], mat.ambient[1], mat.ambient[2]);
    material.diffuse       = glm::vec3(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
    material.specular      = glm::vec3(mat.specular[0], mat.specular[1], mat.specular[2]);
    material.emission      = glm::vec3(mat.emission[0], mat.emission[1], mat.emission[2]);
    material.transmittance = glm::vec3(mat.transmittance[0], mat.transmittance[1], mat.transmittance[2]);
    material.dissolve      = mat.dissolve;
    material.ior           = mat.ior;
    material.shininess     = mat.shininess;
    material.illum         = mat.illum;

    // TODO: handle textures

    mesh.materials.emplace_back(material);
  }
  // If there were none, add a default
  if (mesh.materials.empty()) mesh.materials.emplace_back(material{});

  const tinyobj::attrib_t &attrib = reader.GetAttrib();

  for (const auto &shape : reader.GetShapes()) {

    mesh.vertexes.reserve(shape.mesh.indices.size() + mesh.vertexes.size());
    mesh.indices.reserve(shape.mesh.indices.size() + mesh.indices.size());
    mesh.mat_indices.insert(mesh.mat_indices.end(), shape.mesh.material_ids.begin(), shape.mesh.material_ids.end());

    for (const auto &index : shape.mesh.indices) {
      vertex vertex       = {};
      size_t vertex_index = (size_t) 3 * index.vertex_index;

      vertex.pos = { attrib.vertices[vertex_index + 0], //
                     attrib.vertices[vertex_index + 1], //
                     attrib.vertices[vertex_index + 2] };

      if (!attrib.normals.empty() && index.normal_index >= 0) {
        size_t normal_index = (size_t) 3 * index.normal_index;
        vertex.normal       = { attrib.normals[normal_index + 0], //
                                attrib.normals[normal_index + 1], //
                                attrib.normals[normal_index + 2] };
      }

      if (!attrib.texcoords.empty() && index.texcoord_index >= 0) {
        size_t texture_index = (size_t) 2 * index.texcoord_index;
        vertex.texture       = { attrib.texcoords[texture_index], //
                                 1.0f - attrib.texcoords[texture_index + 1] };
      }

      mesh.vertexes.push_back(vertex);
      mesh.indices.push_back(static_cast<int>(mesh.indices.size()));
    }
  }

  // Fixing material indices
  for (auto &mi : mesh.mat_indices) {
    if (mi < 0 || mi > mesh.materials.size()) mi = 0;
  }

  // Compute normal when no normal were provided.
  if (attrib.normals.empty()) {
    for (size_t i = 0; i < mesh.indices.size(); i += 3) {
      vertex &v0 = mesh.vertexes[mesh.indices[i + 0]];
      vertex &v1 = mesh.vertexes[mesh.indices[i + 1]];
      vertex &v2 = mesh.vertexes[mesh.indices[i + 2]];

      glm::vec3 n = glm::normalize(glm::cross((v1.pos - v0.pos), (v2.pos - v0.pos)));
      v0.normal   = n;
      v1.normal   = n;
      v2.normal   = n;
    }
  }
}

void RayTracer::load_mesh_to_gpu(Mesh &mesh) {

  Context &context = m_context_ref.get();

  // Converting from Srgb to linear
  // TODO: what is going on in here =/
  for (auto &m : mesh.raw.materials) {
    m.ambient  = glm::pow(m.ambient, glm::vec3(2.2f));
    m.diffuse  = glm::pow(m.diffuse, glm::vec3(2.2f));
    m.specular = glm::pow(m.specular, glm::vec3(2.2f));
  }

  mesh.gpu.vertex_count = (u32) mesh.raw.vertexes.size();
  mesh.gpu.index_count  = (u32) mesh.raw.indices.size();

  VkBufferUsageFlags flag = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
  // FIXME: each create buffer is queue submission that is bad
  mesh.gpu.vertex         = context.create_buffer(mesh.raw.vertexes, flag | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
  mesh.gpu.index          = context.create_buffer(mesh.raw.indices, flag | VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
  mesh.gpu.material       = context.create_buffer(mesh.raw.materials, flag | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  mesh.gpu.material_index = context.create_buffer(mesh.raw.mat_indices, flag | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  auto number = m_meshes.size();
  context.set_debug_name(mesh.gpu.vertex.handle, fmt::format("vertex buffer for mesh#{}", number));
  context.set_debug_name(mesh.gpu.index.handle, fmt::format("index buffer for mesh#{}", number));
  context.set_debug_name(mesh.gpu.material.handle, fmt::format("material buffer for mesh#{}", number));
  context.set_debug_name(mesh.gpu.material_index.handle, fmt::format("material_index buffer for mesh#{}", number));

  mesh.description_index = m_description.data.size();

  mesh_description desc       = {};
  desc.txtOffset              = 0;
  desc.vertex_address         = context.get_buffer_device_address(mesh.gpu.vertex.handle);
  desc.index_address          = context.get_buffer_device_address(mesh.gpu.index.handle);
  desc.material_address       = context.get_buffer_device_address(mesh.gpu.material.handle);
  desc.material_index_address = context.get_buffer_device_address(mesh.gpu.material_index.handle);

  m_description.data.push_back(desc);
}

void RayTracer::load_mesh_to_blas(Mesh &mesh) {

  Context const &context = m_context_ref;

  const VkBufferUsageFlags buffer_usage_flags =
      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

  mesh_description desc = m_description.data[mesh.description_index];

  VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
  triangles.sType                    = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;
  triangles.vertexData.deviceAddress = desc.vertex_address;
  triangles.maxVertex                = mesh.gpu.vertex_count;
  triangles.vertexStride             = sizeof(vertex);
  triangles.indexType                = VK_INDEX_TYPE_UINT32;
  triangles.indexData.deviceAddress  = desc.index_address;
  // triangles.transformData = ;

  // The bottom level acceleration structure contains one set of triangles as the input geometry
  VkAccelerationStructureGeometryKHR acceleration_structure_geometry{};
  acceleration_structure_geometry.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  acceleration_structure_geometry.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  acceleration_structure_geometry.flags              = VK_GEOMETRY_OPAQUE_BIT_KHR;
  acceleration_structure_geometry.geometry.triangles = triangles;

  // Get the size requirements for buffers involved in the acceleration structure build process
  VkAccelerationStructureBuildGeometryInfoKHR acceleration_structure_build_geometry_info{};
  acceleration_structure_build_geometry_info.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  acceleration_structure_build_geometry_info.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  acceleration_structure_build_geometry_info.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  acceleration_structure_build_geometry_info.geometryCount = 1;
  acceleration_structure_build_geometry_info.pGeometries   = &acceleration_structure_geometry;

  const u32 triangle_count = mesh.gpu.index_count / 3;

  VkAccelerationStructureBuildSizesInfoKHR acceleration_structure_build_sizes_info{};
  acceleration_structure_build_sizes_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

  assert(vkGetAccelerationStructureBuildSizesKHR != nullptr);
  vkGetAccelerationStructureBuildSizesKHR(
      context.device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &acceleration_structure_build_geometry_info, &triangle_count,
      &acceleration_structure_build_sizes_info
  );

  VkBufferCreateInfo acc_buffer_info{};
  acc_buffer_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  acc_buffer_info.usage       = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  acc_buffer_info.size        = acceleration_structure_build_sizes_info.accelerationStructureSize;
  acc_buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo acc_buffer_alloc = {};
  acc_buffer_alloc.usage                   = VMA_MEMORY_USAGE_GPU_ONLY;

  check(
      vmaCreateBuffer(
          context.vma_allocator(),             //
          &acc_buffer_info, &acc_buffer_alloc, //
          &mesh.blas.buffer.handle, &mesh.blas.buffer.allocation, nullptr
      ),
      "creating buffer for blas"
  );

  context.set_debug_name(mesh.blas.buffer.handle, fmt::format("blas buffer for mesh"));

  // Create the acceleration structure
  VkAccelerationStructureCreateInfoKHR acceleration_structure_create_info{};
  acceleration_structure_create_info.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
  acceleration_structure_create_info.buffer = mesh.blas.buffer.handle;
  acceleration_structure_create_info.size   = acceleration_structure_build_sizes_info.accelerationStructureSize;
  acceleration_structure_create_info.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

  check(vkCreateAccelerationStructureKHR(context.device(), &acceleration_structure_create_info, nullptr, &mesh.blas.handle));

  // The actual build process starts here

  buffer_t           scratch_buffer = {};
  VkBufferCreateInfo scratch_buffer_info{};
  scratch_buffer_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  scratch_buffer_info.usage       = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  scratch_buffer_info.size        = acceleration_structure_build_sizes_info.accelerationStructureSize;
  scratch_buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo scratch_buffer_alloc = {};
  scratch_buffer_alloc.usage                   = VMA_MEMORY_USAGE_CPU_TO_GPU;

  check(
      vmaCreateBuffer(
          context.vma_allocator(),                     //
          &scratch_buffer_info, &scratch_buffer_alloc, //
          &scratch_buffer.handle, &scratch_buffer.allocation, nullptr
      ),
      "creating buffer for blas"
  );

  VkAccelerationStructureBuildGeometryInfoKHR acceleration_build_geometry_info{};
  acceleration_build_geometry_info.sType                     = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  acceleration_build_geometry_info.type                      = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  acceleration_build_geometry_info.flags                     = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  acceleration_build_geometry_info.mode                      = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  acceleration_build_geometry_info.dstAccelerationStructure  = mesh.blas.handle;
  acceleration_build_geometry_info.geometryCount             = 1;
  acceleration_build_geometry_info.pGeometries               = &acceleration_structure_geometry;
  acceleration_build_geometry_info.scratchData.deviceAddress = context.get_buffer_device_address(scratch_buffer.handle);

  VkAccelerationStructureBuildRangeInfoKHR acceleration_structure_build_range_info{};
  acceleration_structure_build_range_info.primitiveCount                                            = triangle_count;
  acceleration_structure_build_range_info.primitiveOffset                                           = 0;
  acceleration_structure_build_range_info.firstVertex                                               = 0;
  acceleration_structure_build_range_info.transformOffset                                           = 0;
  std::array<VkAccelerationStructureBuildRangeInfoKHR*, 1> acceleration_build_structure_range_infos = { &acceleration_structure_build_range_info };
  context.immediate_submit([&](VkCommandBuffer cmd) {
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &acceleration_build_geometry_info, acceleration_build_structure_range_infos.data());
  });

  vmaDestroyBuffer(context.vma_allocator(), scratch_buffer.handle, scratch_buffer.allocation);
}

void RayTracer::load_model(std::string_view mesh_name, glm::mat4 transform) {
  Context const &context = m_context_ref;

  VkAccelerationStructureInstanceKHR instance = {};

  glm::mat3x4 rtxT = glm::transpose(transform);

  VkTransformMatrixKHR transform_matrix = {};
  memcpy(&transform_matrix, glm::value_ptr(transform), sizeof(VkTransformMatrixKHR));

  auto &mesh = m_meshes[std::string{ mesh_name }];
  auto  blas = mesh.blas.handle;

  // Get the bottom acceleration structure's handle, which will be used during the top level acceleration build
  VkAccelerationStructureDeviceAddressInfoKHR acceleration_device_address_info{};
  acceleration_device_address_info.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
  acceleration_device_address_info.accelerationStructure = blas;
  auto device_address                                    = vkGetAccelerationStructureDeviceAddressKHR(context.device(), &acceleration_device_address_info);

  VkAccelerationStructureInstanceKHR blas_instance{};
  blas_instance.transform                              = transform_matrix;
  blas_instance.instanceCustomIndex                    = mesh.description_index;
  blas_instance.mask                                   = 0xFF;
  blas_instance.instanceShaderBindingTableRecordOffset = 0;
  blas_instance.flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
  blas_instance.accelerationStructureReference         = device_address;

  m_blas_instances.push_back(instance);
}

void RayTracer::init_scene() {
  Context &context = m_context_ref;

  // description buffer
  m_description.buffer = context.create_buffer(
      m_description.data, //
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
  );
  m_description.address = context.get_buffer_device_address(m_description.buffer.handle);
  context.set_debug_name(m_description.buffer.handle, "mesh description buffer");

  // create tlas
}
} // namespace whim::vk