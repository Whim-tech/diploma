#include "vk/renderer.hpp"

#include <array>
#include <filesystem>
#include <fstream>

#include "GLFW/glfw3.h"
#include "fmt/format.h"
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "obj_loader.hpp"
#include "tiny_obj_loader.h"

#include "vk/context.hpp"
#include "vk/result.hpp"

#define VK_NO_PROTOTYPES
#include "imgui/imgui.h"
#include "imgui/imgui_impl_vulkan.h"
#include "imgui/imgui_impl_glfw.h"

#include "shader_interface.h"

namespace whim::vk {

VkShaderModule create_shader_module(Context const &context, std::string_view file_path);

void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout);

Renderer::Renderer(Context &context) :
    m_context(context) {

  VkCommandBufferAllocateInfo cmd_buffers_create_info = {};
  cmd_buffers_create_info.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmd_buffers_create_info.commandPool                 = context.command_pool();
  cmd_buffers_create_info.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmd_buffers_create_info.commandBufferCount          = m_frames_count;

  std::vector<VkCommandBuffer> buffers{ (size_t) m_frames_count };
  check(
      vkAllocateCommandBuffers(
          context.device(), &cmd_buffers_create_info,
          buffers.data()
      ), //
      "allocate command buffers"
  );

  VkFenceCreateInfo fence_create_info = {};
  fence_create_info.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_create_info.pNext             = nullptr;
  fence_create_info.flags             = VK_FENCE_CREATE_SIGNALED_BIT;

  VkSemaphoreCreateInfo semaphore_create_info = {};
  semaphore_create_info.sType                 = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  semaphore_create_info.pNext                 = nullptr;
  semaphore_create_info.flags                 = 0;

  for (int i = 0; i < m_frames_count; i += 1) {
    render_frame_data_t data = {};
    check(
        vkCreateSemaphore(
            context.device(), &semaphore_create_info, //
            nullptr, &data.image_available_semaphore
        ),
        "creating image_available_semaphore"
    );

    check(
        vkCreateSemaphore(
            context.device(), &semaphore_create_info, //
            nullptr, &data.render_finished_semaphore
        ),
        "creating render_finished_semaphore"
    );

    check(
        vkCreateFence(context.device(), &fence_create_info, nullptr, &data.in_flight_fence), //
        "creating fence"
    );
    data.command_buffer = buffers[i];

    m_frames_data.push_back(data);
  }

  VkPushConstantRange pc_range = {};
  pc_range.offset              = 0;
  pc_range.size                = sizeof(push_constant_t);
  pc_range.stageFlags          = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
  pipeline_layout_create_info.sType                      = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_create_info.pNext                      = nullptr;
  pipeline_layout_create_info.flags                      = 0;
  pipeline_layout_create_info.setLayoutCount             = 0;
  pipeline_layout_create_info.pSetLayouts                = nullptr;
  pipeline_layout_create_info.pushConstantRangeCount     = 1;
  pipeline_layout_create_info.pPushConstantRanges        = &pc_range;

  check(
      vkCreatePipelineLayout(context.device(), &pipeline_layout_create_info, nullptr, &m_pipeline_layout), //
      "creating pipeline layout"
  );

  auto attributes  = vertex_attributes_description();
  auto description = vertex_description();

  VkPipelineVertexInputStateCreateInfo input_state = {};
  input_state.sType                                = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  input_state.vertexBindingDescriptionCount        = 1;
  input_state.pVertexBindingDescriptions           = &description;
  input_state.vertexAttributeDescriptionCount      = attributes.size();
  input_state.pVertexAttributeDescriptions         = attributes.data();

  VkPipelineInputAssemblyStateCreateInfo input_assembly = {};
  input_assembly.sType                                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  input_assembly.topology                               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  input_assembly.primitiveRestartEnable                 = VK_FALSE;

  VkExtent2D swapchain_size = context.swapchain_extent();

  VkViewport viewport = {};
  viewport.x          = 0.f;
  viewport.y          = 0.f;
  viewport.height     = (float) swapchain_size.height;
  viewport.width      = (float) swapchain_size.width;
  viewport.minDepth   = 0.f;
  viewport.maxDepth   = 1.f;

  VkRect2D scissor = {};
  scissor.offset   = { 0, 0 };
  scissor.extent   = swapchain_size;

  VkPipelineViewportStateCreateInfo viewport_state = {};
  viewport_state.sType                             = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.viewportCount                     = 1;
  viewport_state.pViewports                        = &viewport;
  viewport_state.scissorCount                      = 1;
  viewport_state.pScissors                         = &scissor;

  VkPipelineRasterizationStateCreateInfo rast_state = {};
  rast_state.sType                                  = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rast_state.depthClampEnable                       = VK_FALSE;
  rast_state.rasterizerDiscardEnable                = VK_FALSE;
  rast_state.polygonMode                            = VK_POLYGON_MODE_FILL;
  rast_state.cullMode                               = VK_CULL_MODE_NONE;
  rast_state.frontFace                              = VK_FRONT_FACE_CLOCKWISE;
  rast_state.depthBiasClamp                         = VK_FALSE;
  rast_state.lineWidth                              = 1.f;
  rast_state.depthBiasConstantFactor                = 0.f;
  rast_state.depthBiasClamp                         = 0.f;
  rast_state.depthBiasSlopeFactor                   = 0.f;

  VkPipelineMultisampleStateCreateInfo mult_state = {};
  mult_state.sType                                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  mult_state.rasterizationSamples                 = VK_SAMPLE_COUNT_1_BIT;
  mult_state.sampleShadingEnable                  = VK_FALSE;
  mult_state.minSampleShading                     = 1.f;
  mult_state.minSampleShading                     = 1.0f;
  mult_state.pSampleMask                          = nullptr;
  mult_state.alphaToCoverageEnable                = VK_FALSE;
  mult_state.alphaToOneEnable                     = VK_FALSE;

  VkPipelineDepthStencilStateCreateInfo depth_state = {};
  depth_state.sType                                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depth_state.depthTestEnable                       = VK_TRUE;
  depth_state.depthWriteEnable                      = VK_TRUE;
  depth_state.depthCompareOp                        = VK_COMPARE_OP_LESS;
  depth_state.stencilTestEnable                     = VK_FALSE;
  depth_state.depthBoundsTestEnable                 = VK_FALSE;
  depth_state.minDepthBounds                        = 0.f;
  depth_state.maxDepthBounds                        = 1.f;
  depth_state.front                                 = {};
  depth_state.back                                  = {};

  VkPipelineColorBlendAttachmentState color_attachment = {};
  color_attachment.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  color_attachment.blendEnable         = VK_FALSE;
  color_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
  color_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
  color_attachment.colorBlendOp        = VK_BLEND_OP_ADD;
  color_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  color_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  color_attachment.alphaBlendOp        = VK_BLEND_OP_ADD;

  VkPipelineColorBlendStateCreateInfo blend_state = {};
  blend_state.sType                               = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  blend_state.logicOpEnable                       = VK_FALSE;
  blend_state.attachmentCount                     = 1;
  blend_state.pAttachments                        = &color_attachment;
  blend_state.logicOp                             = VK_LOGIC_OP_COPY;
  blend_state.blendConstants[0]                   = 0.0f;
  blend_state.blendConstants[1]                   = 0.0f;
  blend_state.blendConstants[2]                   = 0.0f;
  blend_state.blendConstants[3]                   = 0.0f;
  blend_state.sType                               = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;

  VkShaderModule vertex_module   = create_shader_module(context, vertex_path);
  VkShaderModule fragment_module = create_shader_module(context, fragment_path);

  VkPipelineShaderStageCreateInfo vert_stage_create_info = {};
  vert_stage_create_info.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vert_stage_create_info.stage                           = VK_SHADER_STAGE_VERTEX_BIT;
  vert_stage_create_info.module                          = vertex_module;
  vert_stage_create_info.pName                           = "main";

  VkPipelineShaderStageCreateInfo frag_stage_create_info = {};
  frag_stage_create_info.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  frag_stage_create_info.stage                           = VK_SHADER_STAGE_FRAGMENT_BIT;
  frag_stage_create_info.module                          = fragment_module;
  frag_stage_create_info.pName                           = "main";

  std::array<VkPipelineShaderStageCreateInfo, 2> shader_stages{ vert_stage_create_info, frag_stage_create_info };

  // Provide information for dynamic rendering
  auto image_format = context.swapchain_image_format();

  VkPipelineRenderingCreateInfoKHR pipeline_create{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR };
  pipeline_create.pNext                   = VK_NULL_HANDLE;
  pipeline_create.colorAttachmentCount    = 1;
  pipeline_create.pColorAttachmentFormats = &image_format;
  pipeline_create.depthAttachmentFormat   = context.swapchain_depth_format();
  pipeline_create.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

  VkGraphicsPipelineCreateInfo pipeline_create_info = {};
  pipeline_create_info.sType                        = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_create_info.pNext                        = &pipeline_create;
  pipeline_create_info.stageCount                   = 2;
  pipeline_create_info.pStages                      = shader_stages.data();
  pipeline_create_info.pVertexInputState            = &input_state;
  pipeline_create_info.pInputAssemblyState          = &input_assembly;
  pipeline_create_info.pViewportState               = &viewport_state;
  pipeline_create_info.pRasterizationState          = &rast_state;
  pipeline_create_info.pMultisampleState            = &mult_state;
  pipeline_create_info.pDepthStencilState           = &depth_state;
  pipeline_create_info.pColorBlendState             = &blend_state;
  pipeline_create_info.layout                       = m_pipeline_layout;
  pipeline_create_info.subpass                      = 0;
  pipeline_create_info.renderPass                   = VK_NULL_HANDLE;
  pipeline_create_info.basePipelineHandle           = nullptr;
  pipeline_create_info.pDynamicState                = nullptr;
  pipeline_create_info.pTessellationState           = nullptr;
  pipeline_create_info.basePipelineIndex            = -1;

  check(
      vkCreateGraphicsPipelines(context.device(), nullptr, 1, &pipeline_create_info, nullptr, &m_pipeline), //
      "creating pipeline"
  );

  vkDestroyShaderModule(context.device(), vertex_module, nullptr);
  vkDestroyShaderModule(context.device(), fragment_module, nullptr);

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

  check(vkCreateDescriptorPool(context.device(), &imgui_pool_info, nullptr, &m_imgui_desc_pool), "");

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
  init_info.DescriptorPool            = m_imgui_desc_pool;
  init_info.MinImageCount             = context.swapchain_image_count();
  init_info.ImageCount                = context.swapchain_image_count();
  init_info.UseDynamicRendering       = true;
  init_info.ColorAttachmentFormat     = context.swapchain_image_format();
  init_info.MSAASamples               = VK_SAMPLE_COUNT_1_BIT;

  ImGui_ImplVulkan_Init(&init_info, VK_NULL_HANDLE);

  ImGui_ImplVulkan_CreateFontsTexture();
}

Renderer::~Renderer() {
  if (m_pipeline != nullptr) {

    Context const &context = m_context;

    vkDeviceWaitIdle(context.device());

    vmaDestroyBuffer(context.vma_allocator(), m_desc_buffer.buffer, m_desc_buffer.allocation);

    for (auto &model : m_model_desc) {
      vmaDestroyBuffer(context.vma_allocator(), model.vertex.buffer, model.vertex.allocation);
      vmaDestroyBuffer(context.vma_allocator(), model.index.buffer, model.index.allocation);
      vmaDestroyBuffer(context.vma_allocator(), model.material.buffer, model.material.allocation);
      vmaDestroyBuffer(context.vma_allocator(), model.material_index.buffer, model.material_index.allocation);
    }

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    vkDestroyDescriptorPool(context.device(), m_imgui_desc_pool, nullptr);

    std::vector<VkCommandBuffer> buffers{ (size_t) m_frames_count };
    for (auto frame_data : m_frames_data) {
      buffers.push_back(frame_data.command_buffer);
      vkDestroyFence(context.device(), frame_data.in_flight_fence, nullptr);
      vkDestroySemaphore(context.device(), frame_data.image_available_semaphore, nullptr);
      vkDestroySemaphore(context.device(), frame_data.render_finished_semaphore, nullptr);
    }

    vkFreeCommandBuffers(
        context.device(),       //
        context.command_pool(), //
        m_frames_count,         //
        buffers.data()
    );

    vkDestroyPipelineLayout(context.device(), m_pipeline_layout, nullptr);
    vkDestroyPipeline(context.device(), m_pipeline, nullptr);
  }
}

void Renderer::load_model(std::string_view const obj_path) {

  Context &context = m_context;

  whim::ObjLoader loader(obj_path);

  // Converting from Srgb to linear
  // TODO: what is going on in here =/
  for (auto &m : loader.materials) {
    m.ambient  = glm::pow(m.ambient, glm::vec3(2.2f));
    m.diffuse  = glm::pow(m.diffuse, glm::vec3(2.2f));
    m.specular = glm::pow(m.specular, glm::vec3(2.2f));
  }

  model_description_t model = {};

  model.vertex_count = loader.vertexes.size();
  model.index_count  = loader.indices.size();

  std::vector<vertex> vertexes{};
  constexpr auto      size = sizeof(vertex);
  vertexes.reserve(loader.vertexes.size());
  for (auto &v : loader.vertexes) {
    vertexes.push_back(vertex{ .pos = v.pos, .u_x = v.texture.x, .normal = v.norm, .u_y = v.texture.y });
  }

  std::vector<material> materials{};
  materials.reserve(loader.materials.size());

  for (auto &m : loader.materials) {
    materials.push_back(material{
        .ambient       = m.ambient,
        .diffuse       = m.diffuse,
        .specular      = m.specular,
        .transmittance = m.transmittance,
        .emission      = m.emission,
        .shininess     = m.shininess,
        .ior           = m.ior,
        .dissolve      = m.dissolve,
        .illum         = m.illum,
        .texture_id    = m.texture_id,
    });
  }

  VkBufferUsageFlags flag = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  // FIXME: each create buffer is queue submission that is bad
  model.vertex         = context.create_buffer(vertexes, flag | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
  model.index          = context.create_buffer(loader.indices, flag | VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
  model.material       = context.create_buffer(materials, flag | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  model.material_index = context.create_buffer(loader.mat_indices, flag | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  auto number = m_model_desc.size();
  context.set_debug_name(model.vertex.buffer, fmt::format("vertex buffer for model#{}", number));
  context.set_debug_name(model.index.buffer, fmt::format("index buffer for model#{}", number));
  context.set_debug_name(model.material.buffer, fmt::format("material buffer for model#{}", number));
  context.set_debug_name(model.material_index.buffer, fmt::format("material_index buffer for model#{}", number));

  object_description desc = {};

  desc.txtOffset              = 0;
  desc.vertex_address         = context.get_buffer_device_address(model.vertex.buffer);
  desc.index_address          = context.get_buffer_device_address(model.index.buffer);
  desc.material_address       = context.get_buffer_device_address(model.material.buffer);
  desc.material_index_address = context.get_buffer_device_address(model.material_index.buffer);

  m_object_desc.push_back(desc);
  m_model_desc.push_back(model);
}

void Renderer::end_load() {
  m_desc_buffer      = m_context.get().create_buffer(m_object_desc, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  m_desc_buffer_addr = m_context.get().get_buffer_device_address(m_desc_buffer.buffer);
}

void Renderer::draw() {
  Context const &context = m_context;

  // imgui new frame
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  // some imgui UI to test
  ImGui::ShowDemoWindow();

  // make imgui calculate internal draw structures
  ImGui::Render();

  constexpr u64 no_timeout = std::numeric_limits<u64>::max();

  render_frame_data_t data = m_frames_data[m_current_frame];
  // wait until the gpu has finished rendering the last frame. Timeout of 1
  // second
  check(vkWaitForFences(context.device(), 1, &data.in_flight_fence, true, no_timeout), "");

  u32 image_index = 0;
  check(
      vkAcquireNextImageKHR(
          context.device(),               //
          context.swapchain(),            //
          no_timeout,                     //
          data.image_available_semaphore, //
          nullptr,                        //
          &image_index
      ),                                  //
      "acquiring next image index from swapchain"
  );

  check(vkResetCommandBuffer(data.command_buffer, 0), "");

  // check(vkResetFences(context.device(), 1, &data.in_flight_fence), "");

  VkCommandBufferBeginInfo begin_info = {};
  begin_info.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags                    = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  begin_info.pInheritanceInfo         = nullptr;

  check(
      vkBeginCommandBuffer(data.command_buffer, &begin_info), //
      "beginning command buffer"
  );

  {

    transition_image(data.command_buffer, context.swapchain_frames()[image_index].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    transition_image(
        data.command_buffer, context.swapchain_frames()[image_index].depth.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
    );

    transition_image(data.command_buffer, context.swapchain_frames()[image_index].image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    VkRenderingAttachmentInfo color_attachment = {};

    color_attachment.sType            = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color_attachment.imageView        = context.swapchain_frames()[image_index].image_view;
    color_attachment.imageLayout      = VK_IMAGE_LAYOUT_GENERAL;
    color_attachment.loadOp           = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp          = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.clearValue.color = {
      {0.0f, 0.5f, 0.0f, 1.0f}
    };

    VkRenderingAttachmentInfo depth_attachment = {};

    depth_attachment.sType                         = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth_attachment.imageView                     = context.swapchain_frames()[image_index].depth.image_view;
    depth_attachment.imageLayout                   = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth_attachment.loadOp                        = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.storeOp                       = VK_ATTACHMENT_STORE_OP_STORE;
    depth_attachment.clearValue.depthStencil.depth = 1.f;

    VkRect2D render_area = {
      .offset = VkOffset2D{0, 0},
        .extent = context.swapchain_extent()
    };

    VkRenderingInfo render_info      = {};
    render_info.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    render_info.layerCount           = 1;
    render_info.colorAttachmentCount = 1;
    render_info.pColorAttachments    = &color_attachment;
    render_info.pDepthAttachment     = &depth_attachment;
    render_info.pStencilAttachment   = nullptr;
    render_info.renderArea           = render_area;

    vkCmdBeginRendering(data.command_buffer, &render_info);

    vkCmdBindPipeline(data.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);

    for (size_t i = 0; i < m_object_desc.size(); i += 1) {

      push_constant_t pc = {};
      pc.obj_index       = i;
      pc.obj_address     = m_desc_buffer_addr;

      auto extent = context.swapchain_extent();
      pc.mvp      = glm::perspectiveRH(45.f, (float) extent.width / (float) extent.height, 0.01f, 100.f) *
               glm::lookAtRH(glm::vec3{ 0.f, 0.f, -3.f }, glm::vec3{ 0.f, 0.f, 0.f }, glm::vec3{ 0.f, 1.f, 0.f });
      pc.mvp = glm::rotate(pc.mvp, glm::radians(float(glfwGetTime()) * 100), glm::vec3{ 0.f, 1.f, 1.f });

      vkCmdPushConstants(data.command_buffer, m_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push_constant_t), &pc);

      vkCmdBindIndexBuffer(data.command_buffer, m_model_desc[i].index.buffer, 0, VK_INDEX_TYPE_UINT32);
      VkDeviceSize offset{ 0 };
      vkCmdBindVertexBuffers(data.command_buffer, 0, 1, &m_model_desc[i].vertex.buffer, &offset);

      vkCmdDrawIndexed(data.command_buffer, m_model_desc[i].index_count, 1, 0, 0, 0);
    }

    vkCmdEndRendering(data.command_buffer);

    VkRenderingAttachmentInfo imgui_color_attachment = {};

    imgui_color_attachment.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    imgui_color_attachment.imageView   = context.swapchain_frames()[image_index].image_view;
    imgui_color_attachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    imgui_color_attachment.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
    imgui_color_attachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo imgui_render_info = {};

    imgui_render_info.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    imgui_render_info.layerCount           = 1;
    imgui_render_info.colorAttachmentCount = 1;
    imgui_render_info.renderArea           = render_area;
    imgui_render_info.pColorAttachments    = &imgui_color_attachment;
    imgui_render_info.pDepthAttachment     = nullptr;
    imgui_render_info.pStencilAttachment   = nullptr;

    vkCmdBeginRendering(data.command_buffer, &imgui_render_info);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), data.command_buffer);

    vkCmdEndRendering(data.command_buffer);
  }

  check(
      vkEndCommandBuffer(data.command_buffer), //
      "ending command buffer"
  );

  VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

  VkSubmitInfo submit_info         = {};
  submit_info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.waitSemaphoreCount   = 1;
  submit_info.pWaitSemaphores      = &data.image_available_semaphore;
  submit_info.pWaitDstStageMask    = &wait_stage;
  submit_info.commandBufferCount   = 1;
  submit_info.pCommandBuffers      = &data.command_buffer;
  submit_info.signalSemaphoreCount = 1;
  submit_info.pSignalSemaphores    = &data.render_finished_semaphore;

  check(vkResetFences(context.device(), 1, &data.in_flight_fence), "reseting fence");

  check(
      vkQueueSubmit(
          context.graphics_queue(), //
          1, &submit_info,          //
          data.in_flight_fence      //
      ),
      fmt::format("submitting {} image to graphics queue on frame{}", image_index, m_current_frame)
  );

  VkSwapchainKHR swapchain = context.swapchain();

  VkPresentInfoKHR present_info   = {};
  present_info.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores    = &data.render_finished_semaphore;
  present_info.swapchainCount     = 1;
  present_info.pSwapchains        = &swapchain;
  present_info.pImageIndices      = &image_index;
  present_info.pResults           = nullptr; // Optional

  check(
      vkQueuePresentKHR(context.present_queue(), &present_info), //
      fmt::format("submitting {} image to present queue in frame{}", image_index, m_current_frame)
  );

  m_current_frame = (m_current_frame + 1) / m_frames_count;
}

VkShaderModule create_shader_module(Context const &context, std::string_view file_path) {

  if (!std::filesystem::exists(file_path)) {
    WERROR("[ERROR] Failed to create ShaderModule: file not found {}", file_path);
    throw std::runtime_error("failed to create shader module");
  }

  std::ifstream file{ file_path.data(), std::ios::binary | std::ios::ate };

  if (!file.is_open()) {
    WERROR("[ERROR] Failed to create ShaderModule: cant open file {}", file_path);
    throw std::runtime_error("failed to create shader module");
  }

  std::size_t       file_size = file.tellg();
  std::vector<char> buffer(file_size);

  file.seekg(0);
  file.read(buffer.data(), file_size);
  file.close();

  VkShaderModuleCreateInfo create_info = {};

  create_info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = buffer.size();
  create_info.pCode    = (u32*) buffer.data();

  VkShaderModule shader_module = nullptr;
  check(
      vkCreateShaderModule(context.device(), &create_info, nullptr, &shader_module), //
      "creating shader module"
  );

  return shader_module;
}

void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout) {
  VkImageMemoryBarrier2 image_barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
  image_barrier.pNext = nullptr;

  image_barrier.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  image_barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
  image_barrier.dstStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  image_barrier.dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;

  image_barrier.oldLayout = currentLayout;
  image_barrier.newLayout = newLayout;

  VkImageAspectFlags aspect_mask = (newLayout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;

  image_barrier.subresourceRange.aspectMask     = aspect_mask;
  image_barrier.subresourceRange.baseMipLevel   = 0;
  image_barrier.subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
  image_barrier.subresourceRange.baseArrayLayer = 0;
  image_barrier.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;
  image_barrier.image                           = image;

  VkDependencyInfo dep_info{};
  dep_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
  dep_info.pNext = nullptr;

  dep_info.imageMemoryBarrierCount = 1;
  dep_info.pImageMemoryBarriers    = &image_barrier;

  vkCmdPipelineBarrier2(cmd, &dep_info);
}

} // namespace whim::vk