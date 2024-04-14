#include "vk/raytracer.hpp"
#include "camera.hpp"
#include "fmt/format.h"
#include "shader.h"
#include <cstddef>
#include <glm/gtc/type_ptr.hpp>

#include <array>
#include <filesystem>

#include <external/stb_image.h>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_vulkan.h"
#include "imgui/imgui_impl_glfw.h"
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"
#include "utility/align.hpp"
#include "vk/context.hpp"
#include "vk/types.hpp"
#include "whim.hpp"

#include <queue>
#include <set>
#include <stdexcept>
#include <tiny_obj_loader.h>
#include <vulkan/vulkan_core.h>

namespace whim::vk {

RayTracer::RayTracer(Context &context, CameraManipulator const &man) :
    m_context_ref(context),
    m_camera_ref(man) {

  create_frame_data();
  init_imgui();

  m_rt_prop.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;

  VkPhysicalDeviceProperties2 prop2{};
  prop2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  prop2.pNext = &m_rt_prop;
  vkGetPhysicalDeviceProperties2(context.physical_device(), &prop2);

  create_storage_image();
  create_uniform_buffer();
  create_offscreen_renderer();
}

RayTracer::~RayTracer() {
  // TODO: check if raytracer is still valid (after move)
  if (true) {
    /*
      ORDER OF CLEAN UP:

      1 - wait for device
      2 - pipeline destroy
      2 - SBT destroy
      2 - desc cleanup
      2 - description buffer destruction
      2 - TLAS cleanup
      2 - BLAS cleanup
      2 - Meshes data cleanup
      2 - Spheres cleanup
      3 - imgui cleanup
      4 - frame data cleanup
      5 - offscreen renderer desctruction
      5 - storage image cleanup
      6 - ubo cleanup
      7 - textures cleanup
    */
    Context const &context = m_context_ref;

    vkDeviceWaitIdle(context.device());

    // pipeline destruction
    vkDestroyPipeline(context.device(), m_pipeline, nullptr);
    vkDestroyPipelineLayout(context.device(), m_pipeline_layout, nullptr);

    // Shader binding table destruction
    vmaDestroyBuffer(context.vma_allocator(), m_sbtb_buffer.handle, m_sbtb_buffer.allocation);

    // shared descriptors
    vkDestroyDescriptorSetLayout(context.device(), m_descriptor.shared.layout, nullptr);
    vkDestroyDescriptorPool(context.device(), m_descriptor.shared.pool, nullptr);

    vmaDestroyBuffer(context.vma_allocator(), m_tlas.buffer.handle, m_tlas.buffer.allocation);
    vkDestroyAccelerationStructureKHR(context.device(), m_tlas.handle, nullptr);

    vmaDestroyBuffer(context.vma_allocator(), m_description.buffer.handle, m_description.buffer.allocation);

    // for (auto const &[k, v] : m_meshes) {
    //   vmaDestroyBuffer(context.vma_allocator(), v.blas.buffer.handle, v.blas.buffer.allocation);
    //   vkDestroyAccelerationStructureKHR(context.device(), v.blas.handle, nullptr);

    //   vmaDestroyBuffer(context.vma_allocator(), v.gpu.index.handle, v.gpu.index.allocation);
    //   vmaDestroyBuffer(context.vma_allocator(), v.gpu.vertex.handle, v.gpu.vertex.allocation);
    //   vmaDestroyBuffer(context.vma_allocator(), v.gpu.material.handle, v.gpu.material.allocation);
    //   vmaDestroyBuffer(context.vma_allocator(), v.gpu.material_index.handle, v.gpu.material_index.allocation);
    // }

    for (auto &blas : m_meshes.blases) {
      vmaDestroyBuffer(context.vma_allocator(), blas.buffer.handle, blas.buffer.allocation);
      vkDestroyAccelerationStructureKHR(context.device(), blas.handle, nullptr);
    }
    vmaDestroyBuffer(context.vma_allocator(), m_meshes.device.pos_buffer.handle, m_meshes.device.pos_buffer.allocation);
    vmaDestroyBuffer(context.vma_allocator(), m_meshes.device.index_buffer.handle, m_meshes.device.index_buffer.allocation);
    vmaDestroyBuffer(context.vma_allocator(), m_meshes.device.normal_buffer.handle, m_meshes.device.normal_buffer.allocation);
    vmaDestroyBuffer(context.vma_allocator(), m_meshes.device.uv_buffer.handle, m_meshes.device.uv_buffer.allocation);
    vmaDestroyBuffer(context.vma_allocator(), m_meshes.device.material_buffer.handle, m_meshes.device.material_buffer.allocation);
    vmaDestroyBuffer(context.vma_allocator(), m_meshes.device.prim_infos.handle, m_meshes.device.prim_infos.allocation);

    // // SPHERES
    // vmaDestroyBuffer(context.vma_allocator(), m_spheres.blas.buffer.handle, m_spheres.blas.buffer.allocation);
    // vkDestroyAccelerationStructureKHR(context.device(), m_spheres.blas.handle, nullptr);

    // vmaDestroyBuffer(context.vma_allocator(), m_spheres.gpu_data.spheres.handle, m_spheres.gpu_data.spheres.allocation);
    // vmaDestroyBuffer(context.vma_allocator(), m_spheres.gpu_data.material.handle, m_spheres.gpu_data.material.allocation);
    // vmaDestroyBuffer(context.vma_allocator(), m_spheres.gpu_data.material_index.handle, m_spheres.gpu_data.material_index.allocation);
    // vmaDestroyBuffer(context.vma_allocator(), m_spheres.gpu_data.aabbs.handle, m_spheres.gpu_data.aabbs.allocation);

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

    vkDestroyDescriptorSetLayout(context.device(), m_offscreen.desc_layout, nullptr);
    vkDestroyDescriptorPool(context.device(), m_offscreen.desc_pool, nullptr);
    vkDestroyPipelineLayout(context.device(), m_offscreen.pipeline_layout, nullptr);
    vkDestroyPipeline(context.device(), m_offscreen.pipeline, nullptr);

    vkDestroySampler(context.device(), m_storage_image.sampler, nullptr);
    vkDestroyImageView(context.device(), m_storage_image.view, nullptr);
    vmaDestroyImage(context.vma_allocator(), m_storage_image.image, m_storage_image.allocation);

    vmaDestroyBuffer(context.vma_allocator(), m_ubo.handle, m_ubo.allocation);

    // TEXTURES

    for (auto &texture : m_textures) {
      if (texture.image.handle != m_default_texture.image.handle) {
        vkDestroySampler(context.device(), texture.sampler, nullptr);
        vkDestroyImageView(context.device(), texture.view, nullptr);
        vmaDestroyImage(context.vma_allocator(), texture.image.handle, texture.image.allocation);
      }
    }
    vkDestroySampler(context.device(), m_default_texture.sampler, nullptr);
    vkDestroyImageView(context.device(), m_default_texture.view, nullptr);
    vmaDestroyImage(context.vma_allocator(), m_default_texture.image.handle, m_default_texture.image.allocation);
  }
}

void RayTracer::update_uniform_buffer(VkCommandBuffer cmd) {

  CameraManipulator const &cam = m_camera_ref;
  // updating ubo
  global_ubo host_ubo{};
  host_ubo.inverse_proj = cam.inverse_proj_matrix();
  host_ubo.inverse_view = cam.inverse_view_matrix();
  host_ubo.proj         = cam.proj_matrix();
  host_ubo.view         = cam.view_matrix();

  VkPipelineStageFlagBits ubo_shader_stages = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;

  // Ensure that the modified UBO is not visible to previous frames.
  VkBufferMemoryBarrier before_barrier{};
  before_barrier.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  before_barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  before_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  before_barrier.buffer        = m_ubo.handle;
  before_barrier.offset        = 0;
  before_barrier.size          = sizeof(host_ubo);
  vkCmdPipelineBarrier(cmd, ubo_shader_stages, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_DEPENDENCY_DEVICE_GROUP_BIT, 0, nullptr, 1, &before_barrier, 0, nullptr);

  // Schedule the host-to-device upload. (hostUBO is copied into the cmd
  // buffer so it is okay to deallocate when the function returns).
  vkCmdUpdateBuffer(cmd, m_ubo.handle, 0, sizeof(global_ubo), &host_ubo);

  // Making sure the updated UBO will be visible.
  VkBufferMemoryBarrier after_barrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
  after_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  after_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  after_barrier.buffer        = m_ubo.handle;
  after_barrier.offset        = 0;
  after_barrier.size          = sizeof(host_ubo);
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, ubo_shader_stages, VK_DEPENDENCY_DEVICE_GROUP_BIT, 0, nullptr, 1, &after_barrier, 0, nullptr);
}

void RayTracer::create_default_texture() {
  // TODO: add error handling

  if (!std::filesystem::exists(default_texture_path)) {
    WERROR("Cant find default texture at path - {}", default_texture_path);
    throw std::runtime_error("failed to find default texture at path");
  }
  int      width = 0, height = 0, channels = 0;
  stbi_uc* stbi_pixels = stbi_load(default_texture_path.data(), &width, &height, &channels, STBI_rgb_alpha);

  if (stbi_pixels == nullptr) {
    WERROR("Failed to read default texture from file: {}", default_texture_path);
    throw std::runtime_error("failed to read default texture at path");
  }
  std::vector<unsigned char> data(stbi_pixels, stbi_pixels + width * height * 4);

  m_default_texture = create_texture(width, height, data, VK_FILTER_NEAREST, VK_FILTER_NEAREST);
  m_textures.push_back(m_default_texture);
}

void RayTracer::load_gltf_scene(std::string_view file_path) {

  Context &context = m_context_ref;

  load_gltf_raw(file_path);

  load_gltf_device();

  m_meshes.blases.reserve(m_meshes.raw.prim_meshes.size());
  for (auto &primitive : m_meshes.raw.primitive_infos) {
    m_meshes.blases.emplace_back(acceleration_structure_t{});
    load_primitive_to_blas(primitive, m_meshes.blases.back());
  }

  m_blas_instances.reserve(m_meshes.raw.nodes.size());

  for (auto &node : m_meshes.raw.nodes) {

    glm::mat3x4          rtxT             = glm::transpose(node.world_matrix);
    VkTransformMatrixKHR transform_matrix = {};
    memcpy(&transform_matrix, glm::value_ptr(rtxT), sizeof(VkTransformMatrixKHR));

    VkAccelerationStructureDeviceAddressInfoKHR acceleration_device_address_info{};
    acceleration_device_address_info.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    acceleration_device_address_info.accelerationStructure = m_meshes.blases[node.primitive_mesh].handle;
    auto device_address                                    = vkGetAccelerationStructureDeviceAddressKHR(context.device(), &acceleration_device_address_info);

    VkAccelerationStructureInstanceKHR instance{};
    instance.transform                              = transform_matrix;
    instance.instanceCustomIndex                    = node.primitive_mesh;
    instance.accelerationStructureReference         = device_address;
    instance.flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instance.mask                                   = 0xFF;
    instance.instanceShaderBindingTableRecordOffset = 0;
    m_blas_instances.emplace_back(instance);
  }

  // create tlas
  create_tlas();

  init_descriptors();

  // init_descriptors();
  create_pipeline();
}

void RayTracer::load_gltf_raw(std::string_view file_path) {
  if (!std::filesystem::exists(file_path)) {
    WERROR("Cant parse gltf scene: file not found - {}", file_path);
    throw std::runtime_error("cant find gltf scene");
  }
  tinygltf::TinyGLTF loader{};
  std::string        warning{};
  std::string        error{};
  tinygltf::Model    tmodel{};

  bool res = loader.LoadASCIIFromFile(&tmodel, &error, &warning, std::string(file_path));

  if (not warning.empty()) {
    WERROR(" GLTF WARNING: {}", warning);
  }

  if (not error.empty()) {
    WERROR("error while loading gltf file {}, message:{}", file_path, error);
  }

  if (not res) {
    WERROR("some how gltf return error code with empty error string, filename:{}", file_path);
  }

  // UPDATING MATERIALS
  m_meshes.raw.materials.reserve(tmodel.materials.size());

  for (auto const &tmat : tmodel.materials) {
    material    m   = {};
    auto const &pbr = tmat.pbrMetallicRoughness;

    m.base_color_factor  = glm::vec3{ pbr.baseColorFactor[0], pbr.baseColorFactor[1], pbr.baseColorFactor[3] };
    m.base_color_texture = pbr.baseColorTexture.index;
    m.roughness_factor   = (float) pbr.roughnessFactor;
    m.metallic_factor    = (float) pbr.metallicFactor;
    m.rm_texture         = pbr.metallicRoughnessTexture.index;
    m.emissive_factor    = tmat.emissiveFactor.size() == 3 ? glm::vec3(tmat.emissiveFactor[0], tmat.emissiveFactor[1], tmat.emissiveFactor[2]) : glm::vec3(0.f);
    m.e_texture          = tmat.emissiveTexture.index;
    m.n_texture          = tmat.normalTexture.index;

    m_meshes.raw.materials.emplace_back(m);
  }
  // add default if there is no materials
  if (m_meshes.raw.materials.empty()) {
    material m           = {};
    m.base_color_texture = -1;
    m.rm_texture         = -1;
    m.n_texture          = -1;
    m.e_texture          = -1;
    m_meshes.raw.materials.push_back(m);
  }

  int         default_scene = tmodel.defaultScene > -1 ? tmodel.defaultScene : 0;
  auto const &tscene        = tmodel.scenes[default_scene];

  std::set<i32> used_meshes;
  WASSERT(not tscene.nodes.empty(), "empty scene =/");

  std::queue<i32> nodes_queue{};
  for (auto node_idx : tscene.nodes) {
    nodes_queue.push(node_idx);
  }
  // BFS for nodes tree and get all unique meshes
  while (not nodes_queue.empty()) {
    i32 node_idx = nodes_queue.front();
    nodes_queue.pop();

    auto const &tnode = tmodel.nodes[node_idx];
    if (tnode.mesh > -1) used_meshes.insert(tnode.mesh);
    for (int child : tnode.children) {
      nodes_queue.push(child);
    }
  }

  u32 index_count     = 0;
  u32 primitive_count = 0;
  for (i32 mesh_idx : used_meshes) {

    std::vector<u32> primitives{};

    auto const &tmesh = tmodel.meshes[mesh_idx];
    for (const auto &tprimitive : tmesh.primitives) {
      if (tprimitive.mode != 4) // Triangle
        continue;

      const auto &pos_accessor = tmodel.accessors[tprimitive.attributes.find("POSITION")->second];
      if (tprimitive.indices > -1) {
        const auto &index_accessor = tmodel.accessors[tprimitive.indices];
        index_count += static_cast<uint32_t>(index_accessor.count);
      } else {
        index_count += static_cast<uint32_t>(pos_accessor.count);
      }
      primitives.emplace_back(primitive_count++);
    }
    m_meshes.raw.mesh_to_primitives[mesh_idx] = std::move(primitives);
  }

  m_meshes.raw.indices.reserve(index_count);

  // Convert all mesh/primitives+ to a single primitive per mesh
  for (const auto &m : used_meshes) {
    auto &tmesh = tmodel.meshes[m];
    for (const auto &tprimitive : tmesh.primitives) {
      primitive_full_info result_mesh{};
      result_mesh.material_index = std::max(0, tprimitive.material);
      result_mesh.vertex_offset  = static_cast<uint32_t>(m_meshes.raw.positions.size());
      result_mesh.index_offset   = static_cast<uint32_t>(m_meshes.raw.indices.size());

      // TODO: primitive cache?...
      // INDICES
      if (tprimitive.indices > -1) {
        const tinygltf::Accessor &index_accessor = tmodel.accessors[tprimitive.indices];

        auto const &index_buffer_view = tmodel.bufferViews[index_accessor.bufferView];
        auto const &index_buffer      = tmodel.buffers[index_buffer_view.buffer];

        auto old_size = m_meshes.raw.indices.size();
        m_meshes.raw.indices.resize(m_meshes.raw.indices.size() + index_accessor.count);

        auto         offset = index_accessor.byteOffset + index_buffer_view.byteOffset;
        const size_t stride = index_accessor.ByteStride(index_buffer_view);
        WASSERT(stride != size_t(-1), "??");

        for (int i = 0; i < (int) index_accessor.count; i += 1) {
          if (index_accessor.componentType == TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT) {
            m_meshes.raw.indices[i + old_size] = *(u32*) (&index_buffer.data[stride * i + offset]);
          } else if (index_accessor.componentType == TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT) {
            m_meshes.raw.indices[i + old_size] = *(u16*) (&index_buffer.data[stride * i + offset]);
          } else if (index_accessor.componentType == TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE) {
            m_meshes.raw.indices[i + old_size] = *(u8*) (&index_buffer.data[stride * i + offset]);
          } else {
            WASSERT(false, "WHAT THE HELL?");
          }
        }

        result_mesh.index_count = index_accessor.count;
      } else {
        // Primitive without indices, creating them
        const auto &taccessor = tmodel.accessors[tprimitive.attributes.find("POSITION")->second];
        for (u32 i = 0; i < taccessor.count; i++)
          m_meshes.raw.indices.push_back(i);
        result_mesh.index_count = taccessor.count;
      }

      // TODO: add cache check
      // VERTICES
      auto const &it_pos_accessor = tprimitive.attributes.find("POSITION");
      WASSERT(it_pos_accessor != tprimitive.attributes.end(), "no position data");

      auto const &pos_accessor = tmodel.accessors[it_pos_accessor->second];
      result_mesh.vertex_count = pos_accessor.count;
      WASSERT(pos_accessor.componentType == TINYGLTF_PARAMETER_TYPE_FLOAT, "");
      WASSERT(pos_accessor.type == TINYGLTF_TYPE_VEC3, "");

      auto const &pos_buffer_view = tmodel.bufferViews[pos_accessor.bufferView];
      auto const &pos_buffer      = tmodel.buffers[pos_buffer_view.buffer];

      auto old_size = m_meshes.raw.positions.size();
      m_meshes.raw.positions.resize(m_meshes.raw.positions.size() + result_mesh.vertex_count);

      auto         offset = pos_accessor.byteOffset + pos_buffer_view.byteOffset;
      const size_t stride = pos_accessor.ByteStride(pos_buffer_view);
      WASSERT(stride != size_t(-1), "??");

      for (int i = 0; i < (int) result_mesh.vertex_count; i += 1) {
        m_meshes.raw.positions[i + old_size] = *(glm::vec3*) (&pos_buffer.data[stride * i + offset]);
      }

      auto const &it_norm_accessor = tprimitive.attributes.find("NORMAL");
      if (it_norm_accessor != tprimitive.attributes.end()) {
        auto const &norm_accessor = tmodel.accessors[it_norm_accessor->second];
        WASSERT(norm_accessor.componentType == TINYGLTF_PARAMETER_TYPE_FLOAT, "");
        WASSERT(norm_accessor.type == TINYGLTF_TYPE_VEC3, "");

        auto const &norm_buffer_view = tmodel.bufferViews[norm_accessor.bufferView];
        auto const &norm_buffer      = tmodel.buffers[norm_buffer_view.buffer];

        auto old_norm_size = m_meshes.raw.normals.size();
        m_meshes.raw.normals.resize(m_meshes.raw.normals.size() + norm_accessor.count);

        auto         norm_offset = norm_accessor.byteOffset + norm_buffer_view.byteOffset;
        const size_t norm_stride = norm_accessor.ByteStride(norm_buffer_view);
        WASSERT(norm_stride != size_t(-1), "??");

        for (int i = 0; i < (int) norm_accessor.count; i += 1) {
          m_meshes.raw.normals[i + old_norm_size] = *(glm::vec3*) (&norm_buffer.data[norm_stride * i + norm_offset]);
        }
      } else {
        // generating normals
        std::vector<glm::vec3> normals(result_mesh.vertex_count);
        for (size_t i = 0; i < index_count; i += 3) {
          uint32_t    ind0 = m_meshes.raw.indices[result_mesh.index_offset + i + 0];
          uint32_t    ind1 = m_meshes.raw.indices[result_mesh.index_offset + i + 1];
          uint32_t    ind2 = m_meshes.raw.indices[result_mesh.index_offset + i + 2];
          const auto &pos0 = m_meshes.raw.positions[ind0 + result_mesh.vertex_offset];
          const auto &pos1 = m_meshes.raw.positions[ind1 + result_mesh.vertex_offset];
          const auto &pos2 = m_meshes.raw.positions[ind2 + result_mesh.vertex_offset];
          const auto  v1   = glm::normalize(pos1 - pos0); // Many normalize, but when objects are really small the
          const auto  v2   = glm::normalize(pos2 - pos0); // cross will go below nv_eps and the normal will be (0,0,0)
          const auto  n    = glm::cross(v1, v2);
          normals[ind0] += n;
          normals[ind1] += n;
          normals[ind2] += n;
        }
        for (auto &n : normals)
          n = glm::normalize(n);
        m_meshes.raw.normals.insert(m_meshes.raw.normals.end(), normals.begin(), normals.end());
      }

      // UVS
      auto const &it_uv_accessor = tprimitive.attributes.find("TEXCOORD_0");
      if (it_uv_accessor != tprimitive.attributes.end()) {
        auto const &uv_accessor = tmodel.accessors[it_uv_accessor->second];
        WASSERT(uv_accessor.componentType == TINYGLTF_PARAMETER_TYPE_FLOAT, "");
        WASSERT(uv_accessor.type == TINYGLTF_TYPE_VEC2, "");

        auto const &uv_buffer_view = tmodel.bufferViews[uv_accessor.bufferView];
        auto const &uv_buffer      = tmodel.buffers[uv_buffer_view.buffer];

        auto old_uv_size = m_meshes.raw.uvs.size();
        m_meshes.raw.uvs.resize(m_meshes.raw.uvs.size() + uv_accessor.count);

        auto         uv_offset = uv_accessor.byteOffset + uv_buffer_view.byteOffset;
        const size_t uv_stride = uv_accessor.ByteStride(uv_buffer_view);
        WASSERT(uv_stride != size_t(-1), "??");

        for (int i = 0; i < (int) uv_accessor.count; i += 1) {
          m_meshes.raw.uvs[i + old_uv_size] = *(glm::vec2*) (&uv_buffer.data[uv_stride * i + uv_offset]);
        }
      } else {
        m_meshes.raw.uvs.insert(m_meshes.raw.uvs.end(), pos_accessor.count, glm::vec2(0, 0));
      }

      m_meshes.raw.primitive_infos.push_back(result_mesh);
    }
  }

  // proccess all nodes

  for (auto node_idx : tscene.nodes) {
    process_node(tmodel, node_idx, glm::scale(glm::mat4{1.f}, glm::vec3{-1.f, 1.f, 1.f}));
  }

  // LOAD ALL TEXTURES (load default one if nothing is found)
  if (tmodel.textures.empty()) {
    create_default_texture();
  }

  for (auto &texture : tmodel.textures) {
    auto &image = tmodel.images[texture.source];

    WASSERT(image.bits == 8, "TODO");

    m_textures.push_back(create_texture(image.width, image.height, image.image, VK_FILTER_NEAREST, VK_FILTER_NEAREST));
  }
}

void RayTracer::process_node(const tinygltf::Model &tmodel, int &node_idx, const glm::mat4 &parent_matrix) {
  const auto &tnode = tmodel.nodes[node_idx];

  glm::mat4 translation_matrix = 1;
  glm::mat4 scale_matrix       = 1;
  glm::mat4 rotation_matrix    = 1;
  glm::mat4 node_matrix        = 1;
  glm::quat rotation{};

  if (not tnode.translation.empty())
    translation_matrix = glm::translate(glm::mat4(1), glm::vec3(tnode.translation[0], tnode.translation[1], tnode.translation[2]));
  if (not tnode.scale.empty()) {
    scale_matrix = glm::scale(glm::mat4(1), glm::vec3(tnode.scale[0], tnode.scale[1], tnode.scale[2]));
  }
  if (not tnode.rotation.empty()) {
    rotation        = glm::make_quat(tnode.rotation.data());
    rotation_matrix = glm::mat4_cast(rotation);
  }
  if (!tnode.matrix.empty()) {
    node_matrix = glm::make_mat4(tnode.matrix.data());
  }

  glm::mat4 matrix       = translation_matrix * rotation_matrix * scale_matrix * node_matrix;
  glm::mat4 world_matrix = parent_matrix * matrix;

  if (tnode.mesh > -1) {
    const auto &meshes = m_meshes.raw.mesh_to_primitives[tnode.mesh]; // A mesh could have many primitives
    for (const auto &mesh : meshes) {
      node node;
      node.primitive_mesh = mesh;
      node.world_matrix   = world_matrix;
      m_meshes.raw.nodes.emplace_back(node);
    }
  }

  for (auto child : tnode.children) {
    process_node(tmodel, child, world_matrix);
  }
}

void RayTracer::load_gltf_device() {
  Context &context = m_context_ref;

  auto flags                      = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  m_meshes.device.pos_buffer      = context.create_buffer(m_meshes.raw.positions, flags | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_meshes.device.index_buffer    = context.create_buffer(m_meshes.raw.indices, flags | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_meshes.device.normal_buffer   = context.create_buffer(m_meshes.raw.normals, flags);
  m_meshes.device.uv_buffer       = context.create_buffer(m_meshes.raw.uvs, flags);
  m_meshes.device.material_buffer = context.create_buffer(m_meshes.raw.materials, flags);

  m_meshes.raw.prim_meshes.reserve(m_meshes.raw.primitive_infos.size());
  for (auto &info : m_meshes.raw.primitive_infos) {
    m_meshes.raw.prim_meshes.emplace_back(primitive_shader_info                           //
                                          {
                                              .index_offset   = info.index_offset,        //
                                              .vertex_offset  = info.vertex_offset,       //
                                              .material_index = (int) info.material_index //
                                          });
  }
  m_meshes.device.prim_infos = context.create_buffer(m_meshes.raw.prim_meshes, flags);

  scene_description scene{};
  scene.pos_address       = context.get_buffer_device_address(m_meshes.device.pos_buffer.handle);
  scene.index_address     = context.get_buffer_device_address(m_meshes.device.index_buffer.handle);
  scene.normal_address    = context.get_buffer_device_address(m_meshes.device.normal_buffer.handle);
  scene.uv_address        = context.get_buffer_device_address(m_meshes.device.uv_buffer.handle);
  scene.material_address  = context.get_buffer_device_address(m_meshes.device.material_buffer.handle);
  scene.prim_info_address = context.get_buffer_device_address(m_meshes.device.prim_infos.handle);

  m_description.data.emplace_back(scene);

  m_description.buffer = context.create_buffer(m_description.data, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  context.set_debug_name(m_meshes.device.pos_buffer.handle, "position");
  context.set_debug_name(m_meshes.device.index_buffer.handle, "index");
  context.set_debug_name(m_meshes.device.normal_buffer.handle, "normal");
  context.set_debug_name(m_meshes.device.uv_buffer.handle, "uv");
  context.set_debug_name(m_meshes.device.material_buffer.handle, "material");
  context.set_debug_name(m_meshes.device.prim_infos.handle, "primitive infos");
  context.set_debug_name(m_description.buffer.handle, "scene description");
}

void RayTracer::load_primitive_to_blas(primitive_full_info &primitive, acceleration_structure_t &blas) {
  Context const &context = m_context_ref;

  VkDeviceAddress vertex_address      = context.get_buffer_device_address(m_meshes.device.pos_buffer.handle);
  VkDeviceAddress index_address       = context.get_buffer_device_address(m_meshes.device.index_buffer.handle);
  u32             max_primitive_count = primitive.index_count / 3;

  // Describe buffer as array of VertexObj.
  VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
  triangles.sType                    = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;
  triangles.vertexData.deviceAddress = vertex_address;
  triangles.vertexStride             = sizeof(glm::vec3);
  triangles.indexType                = VK_INDEX_TYPE_UINT32;
  triangles.indexData.deviceAddress  = index_address;
  triangles.maxVertex                = primitive.vertex_count;

  // The bottom level acceleration structure contains one set of triangles as the input geometry
  VkAccelerationStructureGeometryKHR acceleration_structure_geometry{};
  acceleration_structure_geometry.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  acceleration_structure_geometry.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  acceleration_structure_geometry.flags              = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
  acceleration_structure_geometry.geometry.triangles = triangles;

  // Get the size requirements for buffers involved in the acceleration structure build process
  VkAccelerationStructureBuildGeometryInfoKHR acceleration_structure_build_geometry_info{};
  acceleration_structure_build_geometry_info.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  acceleration_structure_build_geometry_info.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  acceleration_structure_build_geometry_info.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  acceleration_structure_build_geometry_info.geometryCount = 1;
  acceleration_structure_build_geometry_info.pGeometries   = &acceleration_structure_geometry;

  VkAccelerationStructureBuildSizesInfoKHR acceleration_structure_build_sizes_info{};
  acceleration_structure_build_sizes_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

  vkGetAccelerationStructureBuildSizesKHR(
      context.device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &acceleration_structure_build_geometry_info, &max_primitive_count,
      &acceleration_structure_build_sizes_info
  );

  VkBufferCreateInfo acc_buffer_info{};
  acc_buffer_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  acc_buffer_info.usage       = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  acc_buffer_info.size        = acceleration_structure_build_sizes_info.accelerationStructureSize;
  acc_buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo acc_buffer_alloc{};
  acc_buffer_alloc.usage = VMA_MEMORY_USAGE_GPU_ONLY;

  check(
      vmaCreateBuffer(
          context.vma_allocator(),             //
          &acc_buffer_info, &acc_buffer_alloc, //
          &blas.buffer.handle, &blas.buffer.allocation, nullptr
      ),
      "creating buffer for blas"
  );
  context.set_debug_name(blas.buffer.handle, fmt::format("blas buffer for mesh"));

  // Create the acceleration structure
  VkAccelerationStructureCreateInfoKHR acceleration_structure_create_info{};
  acceleration_structure_create_info.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
  acceleration_structure_create_info.buffer = blas.buffer.handle;
  acceleration_structure_create_info.size   = acceleration_structure_build_sizes_info.accelerationStructureSize;
  acceleration_structure_create_info.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

  check(vkCreateAccelerationStructureKHR(context.device(), &acceleration_structure_create_info, nullptr, &blas.handle));

  // The actual build process starts here

  buffer_t           scratch_buffer = {};
  VkBufferCreateInfo scratch_buffer_info{};
  scratch_buffer_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  scratch_buffer_info.usage       = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  scratch_buffer_info.size        = acceleration_structure_build_sizes_info.buildScratchSize;
  scratch_buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo scratch_buffer_alloc = {};
  scratch_buffer_alloc.usage                   = VMA_MEMORY_USAGE_CPU_TO_GPU;

  check(
      vmaCreateBuffer(
          context.vma_allocator(),                     //
          &scratch_buffer_info, &scratch_buffer_alloc, //
          &scratch_buffer.handle, &scratch_buffer.allocation, nullptr
      ),
      "creating scratch buffer for blas"
  );

  VkAccelerationStructureBuildGeometryInfoKHR acceleration_build_geometry_info{};
  acceleration_build_geometry_info.sType                     = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  acceleration_build_geometry_info.type                      = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  acceleration_build_geometry_info.flags                     = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  acceleration_build_geometry_info.mode                      = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  acceleration_build_geometry_info.dstAccelerationStructure  = blas.handle;
  acceleration_build_geometry_info.geometryCount             = 1;
  acceleration_build_geometry_info.pGeometries               = &acceleration_structure_geometry;
  acceleration_build_geometry_info.scratchData.deviceAddress = context.get_buffer_device_address(scratch_buffer.handle);

  VkAccelerationStructureBuildRangeInfoKHR acceleration_structure_build_range_info;
  acceleration_structure_build_range_info.firstVertex     = primitive.vertex_offset;
  acceleration_structure_build_range_info.primitiveCount  = max_primitive_count;
  acceleration_structure_build_range_info.primitiveOffset = primitive.index_offset * sizeof(uint32_t);
  acceleration_structure_build_range_info.transformOffset = 0;

  std::array<VkAccelerationStructureBuildRangeInfoKHR*, 1> acceleration_build_structure_range_infos = { &acceleration_structure_build_range_info };

  context.immediate_submit([&](VkCommandBuffer cmd) {
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &acceleration_build_geometry_info, acceleration_build_structure_range_infos.data());
  });

  vmaDestroyBuffer(context.vma_allocator(), scratch_buffer.handle, scratch_buffer.allocation);
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

  // --------------- UPDATING UBO
  update_uniform_buffer(frame.cmd);

  // ------------ DRAWING IN THERE -----------------
  vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipeline);

  std::array<VkDescriptorSet, 1> sets{ m_descriptor.shared.set };
  vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipeline_layout, 0, (u32) sets.size(), sets.data(), 0, nullptr);
  push_constant_t pc{};
  pc.mvp   = glm::mat4{ 1.f };
  pc.frame = m_shader_frame;

  vkCmdPushConstants(
      frame.cmd, m_pipeline_layout,
      VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_CALLABLE_BIT_KHR, 0,
      sizeof(push_constant_t), &pc
  );

  VkExtent2D extent = context.swapchain_extent();
  vkCmdTraceRaysKHR(frame.cmd, &m_gen_region, &m_miss_region, &m_hit_region, &m_call_region, extent.width, extent.height, 1);

  // -------- RENDERING STORAGE IMAGE ---------------------
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

  VkRenderingInfo render_info      = {};
  render_info.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
  render_info.layerCount           = 1;
  render_info.colorAttachmentCount = 1;
  render_info.pColorAttachments    = &color_attachment;
  render_info.pDepthAttachment     = nullptr;
  render_info.pStencilAttachment   = nullptr;
  render_info.renderArea           = render_area;

  vkCmdBeginRendering(frame.cmd, &render_info);

  vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_offscreen.pipeline);
  vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_offscreen.pipeline_layout, 0, 1, &m_offscreen.desc_set, 0, nullptr);
  vkCmdDraw(frame.cmd, 3, 1, 0, 0);

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
  m_shader_frame  = std::max(m_shader_frame + 1, m_maxFrames);
}

void RayTracer::create_storage_image() {
  Context &context = m_context_ref;

  VkExtent2D extent = context.swapchain_extent();

  m_storage_image.width  = extent.width;
  m_storage_image.height = extent.height;
  m_storage_image.format = VK_FORMAT_R32G32B32A32_SFLOAT;
  m_storage_image.type   = VK_IMAGE_TYPE_2D;

  VkImageCreateInfo image_create_info{};
  image_create_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_create_info.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  image_create_info.imageType     = m_storage_image.type;
  image_create_info.format        = m_storage_image.format;
  image_create_info.extent.width  = m_storage_image.width;
  image_create_info.extent.height = m_storage_image.height;
  image_create_info.extent.depth  = 1;
  image_create_info.mipLevels     = 1;
  image_create_info.arrayLayers   = 1;
  image_create_info.samples       = VK_SAMPLE_COUNT_1_BIT;
  image_create_info.tiling        = VK_IMAGE_TILING_OPTIMAL;
  image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  VmaAllocationCreateInfo image_alloc_info{};
  image_alloc_info.usage = VMA_MEMORY_USAGE_AUTO;

  check(
      vmaCreateImage(context.vma_allocator(), &image_create_info, &image_alloc_info, &m_storage_image.image, &m_storage_image.allocation, nullptr),
      "creating storage image"
  );

  VkImageViewCreateInfo image_view_create_info{};
  image_view_create_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  image_view_create_info.image                           = m_storage_image.image;
  image_view_create_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
  image_view_create_info.format                          = m_storage_image.format;
  image_view_create_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  image_view_create_info.subresourceRange.baseMipLevel   = 0;
  image_view_create_info.subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
  image_view_create_info.subresourceRange.baseArrayLayer = 0;
  image_view_create_info.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;

  check(
      vkCreateImageView(context.device(), &image_view_create_info, nullptr, &m_storage_image.view), //
      "creating image view for storage image"
  );

  VkSamplerCreateInfo sampler_info{};
  sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;

  check(
      vkCreateSampler(context.device(), &sampler_info, nullptr, &m_storage_image.sampler), //
      "creating sampler for storage image"
  );

  context.set_debug_name(m_storage_image.image, "storage image");
  context.set_debug_name(m_storage_image.view, "storage image view");

  context.immediate_submit([&](VkCommandBuffer cmd) {
    context.transition_image(cmd, m_storage_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
  });
}

void RayTracer::create_uniform_buffer() {
  Context &context = m_context_ref;

  VkBufferCreateInfo buffer_info{};
  buffer_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.usage       = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  buffer_info.size        = sizeof(global_ubo);
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo alloc_info = {};
  alloc_info.usage                   = VMA_MEMORY_USAGE_AUTO;
  alloc_info.flags                   = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  alloc_info.preferredFlags          = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  check(
      vmaCreateBuffer(context.vma_allocator(), &buffer_info, &alloc_info, &m_ubo.handle, &m_ubo.allocation, nullptr), //
      "allocating uniform buffer"
  );
  context.set_debug_name(m_ubo.handle, "uniform buffer");
}

void RayTracer::create_offscreen_renderer() {
  Context &context = m_context_ref;

  // DESCRIPTOR POOL
  VkDescriptorPoolSize pool_size{};
  pool_size.descriptorCount = 1;
  pool_size.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

  VkDescriptorPoolCreateInfo desc_pool_info{};
  desc_pool_info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  desc_pool_info.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  desc_pool_info.maxSets       = 1;
  desc_pool_info.poolSizeCount = 1;
  desc_pool_info.pPoolSizes    = &pool_size;

  check(
      vkCreateDescriptorPool(context.device(), &desc_pool_info, nullptr, &m_offscreen.desc_pool), //
      "allocation descriptor pool for offscreen renderer"
  );

  // DESCRIPTOR SET LAYOUT
  VkDescriptorSetLayoutBinding image_binding{};
  image_binding.binding         = 0;
  image_binding.descriptorCount = 1;
  image_binding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  image_binding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutCreateInfo desc_layout_info{};
  desc_layout_info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  desc_layout_info.bindingCount = 1;
  desc_layout_info.pBindings    = &image_binding;

  check(
      vkCreateDescriptorSetLayout(context.device(), &desc_layout_info, nullptr, &m_offscreen.desc_layout), //
      "creating descriptor set layout for offscreen renderer"
  );

  // DESCRIPTOR SET
  VkDescriptorSetAllocateInfo set_allocate_info{};
  set_allocate_info.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  set_allocate_info.descriptorPool     = m_offscreen.desc_pool;
  set_allocate_info.descriptorSetCount = 1;
  set_allocate_info.pSetLayouts        = &m_offscreen.desc_layout;

  check(
      vkAllocateDescriptorSets(context.device(), &set_allocate_info, &m_offscreen.desc_set), //
      "allocating descriptor set for offscreen rendering"
  );

  // UPDATING DESC SET
  VkDescriptorImageInfo image_descriptor{};
  image_descriptor.imageView   = m_storage_image.view;
  image_descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  image_descriptor.sampler     = m_storage_image.sampler;

  VkWriteDescriptorSet image_write{};
  image_write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  image_write.dstSet          = m_offscreen.desc_set;
  image_write.dstBinding      = 0;
  image_write.descriptorCount = 1;
  image_write.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  image_write.pImageInfo      = &image_descriptor;

  vkUpdateDescriptorSets(context.device(), 1, &image_write, 0, nullptr);

  // PIPELINE LAYOUT
  VkPipelineLayoutCreateInfo pipeline_layout_create_info{};
  pipeline_layout_create_info.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_create_info.setLayoutCount = 1;
  pipeline_layout_create_info.pSetLayouts    = &m_offscreen.desc_layout;

  check(
      vkCreatePipelineLayout(context.device(), &pipeline_layout_create_info, nullptr, &m_offscreen.pipeline_layout), //
      "creating pipeline layout for offscreen renderer"
  );

  VkPipelineVertexInputStateCreateInfo input_state{};
  input_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  VkPipelineInputAssemblyStateCreateInfo input_assembly{};
  input_assembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  input_assembly.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  input_assembly.primitiveRestartEnable = VK_FALSE;

  VkExtent2D swapchain_size = context.swapchain_extent();

  VkViewport viewport{};
  viewport.x        = 0.f;
  viewport.y        = 0.f;
  viewport.height   = (float) swapchain_size.height;
  viewport.width    = (float) swapchain_size.width;
  viewport.minDepth = 0.f;
  viewport.maxDepth = 1.f;

  VkRect2D scissor{};
  scissor.offset = { 0, 0 };
  scissor.extent = swapchain_size;

  VkPipelineViewportStateCreateInfo viewport_state{};
  viewport_state.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.viewportCount = 1;
  viewport_state.pViewports    = &viewport;
  viewport_state.scissorCount  = 1;
  viewport_state.pScissors     = &scissor;

  VkPipelineRasterizationStateCreateInfo rast_state{};
  rast_state.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rast_state.depthClampEnable        = VK_FALSE;
  rast_state.rasterizerDiscardEnable = VK_FALSE;
  rast_state.polygonMode             = VK_POLYGON_MODE_FILL;
  rast_state.cullMode                = VK_CULL_MODE_NONE;
  rast_state.frontFace               = VK_FRONT_FACE_CLOCKWISE;
  rast_state.depthBiasClamp          = VK_FALSE;
  rast_state.lineWidth               = 1.f;
  rast_state.depthBiasConstantFactor = 0.f;
  rast_state.depthBiasClamp          = 0.f;
  rast_state.depthBiasSlopeFactor    = 0.f;

  VkPipelineMultisampleStateCreateInfo mult_state{};
  mult_state.sType                 = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  mult_state.rasterizationSamples  = VK_SAMPLE_COUNT_1_BIT;
  mult_state.sampleShadingEnable   = VK_FALSE;
  mult_state.minSampleShading      = 1.f;
  mult_state.minSampleShading      = 1.0f;
  mult_state.pSampleMask           = nullptr;
  mult_state.alphaToCoverageEnable = VK_FALSE;
  mult_state.alphaToOneEnable      = VK_FALSE;

  VkPipelineDepthStencilStateCreateInfo depth_state{};
  depth_state.sType           = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depth_state.depthTestEnable = VK_FALSE;

  VkPipelineColorBlendAttachmentState color_attachment{};
  color_attachment.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  color_attachment.blendEnable         = VK_FALSE;
  color_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
  color_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
  color_attachment.colorBlendOp        = VK_BLEND_OP_ADD;
  color_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  color_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  color_attachment.alphaBlendOp        = VK_BLEND_OP_ADD;

  VkPipelineColorBlendStateCreateInfo blend_state{};
  blend_state.sType             = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  blend_state.logicOpEnable     = VK_FALSE;
  blend_state.attachmentCount   = 1;
  blend_state.pAttachments      = &color_attachment;
  blend_state.logicOp           = VK_LOGIC_OP_COPY;
  blend_state.blendConstants[0] = 0.0f;
  blend_state.blendConstants[1] = 0.0f;
  blend_state.blendConstants[2] = 0.0f;
  blend_state.blendConstants[3] = 0.0f;

  VkShaderModule vertex_module   = context.create_shader_module("./spv/offscreen.vert.spv");
  VkShaderModule fragment_module = context.create_shader_module("./spv/offscreen.frag.spv");

  VkPipelineShaderStageCreateInfo vert_stage_create_info{};
  vert_stage_create_info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vert_stage_create_info.stage  = VK_SHADER_STAGE_VERTEX_BIT;
  vert_stage_create_info.module = vertex_module;
  vert_stage_create_info.pName  = "main";

  VkPipelineShaderStageCreateInfo frag_stage_create_info{};
  frag_stage_create_info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  frag_stage_create_info.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
  frag_stage_create_info.module = fragment_module;
  frag_stage_create_info.pName  = "main";

  std::array<VkPipelineShaderStageCreateInfo, 2> shader_stages{ vert_stage_create_info, frag_stage_create_info };

  VkFormat format = context.swapchain_image_format();

  VkPipelineRenderingCreateInfoKHR pipeline_create{};
  pipeline_create.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
  pipeline_create.pNext                   = VK_NULL_HANDLE;
  pipeline_create.colorAttachmentCount    = 1;
  pipeline_create.pColorAttachmentFormats = &format;
  pipeline_create.depthAttachmentFormat   = VK_FORMAT_UNDEFINED;
  pipeline_create.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

  VkGraphicsPipelineCreateInfo pipeline_create_info{};
  pipeline_create_info.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_create_info.pNext               = &pipeline_create;
  pipeline_create_info.stageCount          = (u32) shader_stages.size();
  pipeline_create_info.pStages             = shader_stages.data();
  pipeline_create_info.pVertexInputState   = &input_state;
  pipeline_create_info.pInputAssemblyState = &input_assembly;
  pipeline_create_info.pViewportState      = &viewport_state;
  pipeline_create_info.pRasterizationState = &rast_state;
  pipeline_create_info.pMultisampleState   = &mult_state;
  pipeline_create_info.pDepthStencilState  = &depth_state;
  pipeline_create_info.pColorBlendState    = &blend_state;
  pipeline_create_info.layout              = m_offscreen.pipeline_layout;
  pipeline_create_info.subpass             = 0;
  pipeline_create_info.renderPass          = VK_NULL_HANDLE;
  pipeline_create_info.basePipelineHandle  = nullptr;
  pipeline_create_info.pDynamicState       = nullptr;
  pipeline_create_info.pTessellationState  = nullptr;
  pipeline_create_info.basePipelineIndex   = -1;

  check(
      vkCreateGraphicsPipelines(context.device(), nullptr, 1, &pipeline_create_info, nullptr, &m_offscreen.pipeline), //
      "creating offscreen pipeline"
  );

  vkDestroyShaderModule(context.device(), vertex_module, nullptr);
  vkDestroyShaderModule(context.device(), fragment_module, nullptr);
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
  imgui_pool_info.poolSizeCount              = (u32) std::size(pool_sizes);
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

void RayTracer::init_descriptors() {
  Context &context = m_context_ref;

  std::array<VkDescriptorPoolSize, 5> shader_pool_sizes = {
    VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,                       1},
    VkDescriptorPoolSize{             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,                       1},
    VkDescriptorPoolSize{            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,                       1},
    VkDescriptorPoolSize{            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,                       3},
    VkDescriptorPoolSize{    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, (u32) m_textures.size()},
  };

  // shared descriptor set creations
  VkDescriptorPoolCreateInfo shared_pool_info{};
  shared_pool_info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  shared_pool_info.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  shared_pool_info.maxSets       = 1;
  shared_pool_info.poolSizeCount = (u32) shader_pool_sizes.size();
  shared_pool_info.pPoolSizes    = shader_pool_sizes.data();

  check(
      vkCreateDescriptorPool(context.device(), &shared_pool_info, nullptr, &m_descriptor.shared.pool), //
      "allocation descriptor pool for shared data"
  );

  // TLAS
  VkDescriptorSetLayoutBinding tlas_layout_binding{};
  tlas_layout_binding.binding         = SharedBindings::TLAS;
  tlas_layout_binding.descriptorCount = 1;
  tlas_layout_binding.descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  tlas_layout_binding.stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;

  // STORAGE IMAGE
  VkDescriptorSetLayoutBinding storage_image_binding{};
  storage_image_binding.binding         = SharedBindings::StorageImage;
  storage_image_binding.descriptorCount = 1;
  storage_image_binding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  storage_image_binding.stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

  // UNIFORM BUFFER
  VkDescriptorSetLayoutBinding uniform_buffer_binding{};
  uniform_buffer_binding.binding         = SharedBindings::UniformBuffer;
  uniform_buffer_binding.descriptorCount = 1;
  uniform_buffer_binding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  uniform_buffer_binding.stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

  // OBJECT DESCRIPTIONS
  VkDescriptorSetLayoutBinding description_buffer_binding{};
  description_buffer_binding.binding         = SharedBindings::SceneDescriptions;
  description_buffer_binding.descriptorCount = 1;
  description_buffer_binding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  description_buffer_binding.stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;

  // TEXTURES
  VkDescriptorSetLayoutBinding textures_binding{};
  textures_binding.binding         = SharedBindings::Textures;
  textures_binding.descriptorCount = (u32) m_textures.size();
  textures_binding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  textures_binding.stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;

  // // SPHERES
  // VkDescriptorSetLayoutBinding spheres_buffer_binding{};
  // spheres_buffer_binding.binding         = SharedBindings::Spheres;
  // spheres_buffer_binding.descriptorCount = 1;
  // spheres_buffer_binding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  // spheres_buffer_binding.stageFlags      = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR;

  // PRIMITIVES INFO
  VkDescriptorSetLayoutBinding primitives_buffer_binding{};
  primitives_buffer_binding.binding         = SharedBindings::Primitives;
  primitives_buffer_binding.descriptorCount = 1;
  primitives_buffer_binding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  primitives_buffer_binding.stageFlags      = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR;

  std::array<VkDescriptorSetLayoutBinding, 6> bindings = //
      {
        tlas_layout_binding,                             //
        storage_image_binding,                           //
        uniform_buffer_binding,                          //
        description_buffer_binding,                      //
        textures_binding,                                //
        primitives_buffer_binding
        // spheres_buffer_binding ,                          //
      };

  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = (u32) bindings.size();
  layout_info.pBindings    = bindings.data();

  check(
      vkCreateDescriptorSetLayout(context.device(), &layout_info, nullptr, &m_descriptor.shared.layout), //
      "creating descriptor set layout for shared data"
  );

  VkDescriptorSetAllocateInfo set_allocate_info{};
  set_allocate_info.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  set_allocate_info.descriptorPool     = m_descriptor.shared.pool;
  set_allocate_info.descriptorSetCount = 1;
  set_allocate_info.pSetLayouts        = &m_descriptor.shared.layout;

  check(
      vkAllocateDescriptorSets(context.device(), &set_allocate_info, &m_descriptor.shared.set), //
      "allocating shared descriptor set"
  );

  VkWriteDescriptorSetAccelerationStructureKHR as_descriptor_structure{};
  as_descriptor_structure.sType                      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
  as_descriptor_structure.accelerationStructureCount = 1;
  as_descriptor_structure.pAccelerationStructures    = &m_tlas.handle;

  VkWriteDescriptorSet as_write{};
  as_write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  as_write.dstSet          = m_descriptor.shared.set;
  as_write.dstBinding      = SharedBindings::TLAS;
  as_write.descriptorCount = 1;
  as_write.descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  as_write.pNext           = &as_descriptor_structure;

  VkDescriptorImageInfo image_descriptor{};
  image_descriptor.imageView   = m_storage_image.view;
  image_descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

  VkWriteDescriptorSet image_write{};
  image_write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  image_write.dstSet          = m_descriptor.shared.set;
  image_write.dstBinding      = SharedBindings::StorageImage;
  image_write.descriptorCount = 1;
  image_write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  image_write.pImageInfo      = &image_descriptor;

  VkDescriptorBufferInfo ubo_descriptor{};
  ubo_descriptor.buffer = m_ubo.handle;
  ubo_descriptor.offset = 0;
  ubo_descriptor.range  = VK_WHOLE_SIZE;

  VkWriteDescriptorSet ubo_write{};
  ubo_write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  ubo_write.dstSet          = m_descriptor.shared.set;
  ubo_write.dstBinding      = SharedBindings::UniformBuffer;
  ubo_write.descriptorCount = 1;
  ubo_write.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  ubo_write.pBufferInfo     = &ubo_descriptor;

  VkDescriptorBufferInfo scene_descriptor{};
  scene_descriptor.buffer = m_description.buffer.handle;
  scene_descriptor.offset = 0;
  scene_descriptor.range  = VK_WHOLE_SIZE;

  VkWriteDescriptorSet scene_write{};
  scene_write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  scene_write.dstSet          = m_descriptor.shared.set;
  scene_write.dstBinding      = SharedBindings::SceneDescriptions;
  scene_write.descriptorCount = 1;
  scene_write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  scene_write.pBufferInfo     = &scene_descriptor;

  VkDescriptorBufferInfo primitive_descriptor{};
  primitive_descriptor.buffer = m_meshes.device.prim_infos.handle;
  primitive_descriptor.offset = 0;
  primitive_descriptor.range  = VK_WHOLE_SIZE;

  VkWriteDescriptorSet primitive_write{};
  primitive_write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  primitive_write.dstSet          = m_descriptor.shared.set;
  primitive_write.dstBinding      = SharedBindings::Primitives;
  primitive_write.descriptorCount = 1;
  primitive_write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  primitive_write.pBufferInfo     = &primitive_descriptor;

  std::vector<VkDescriptorImageInfo> textures_info{ (size_t) m_textures.size(), VkDescriptorImageInfo{} };
  for (int i = 0; i < m_textures.size(); i += 1) {
    textures_info[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    textures_info[i].imageView   = m_textures[i].view;
    textures_info[i].sampler     = m_textures[i].sampler;
  }
  VkWriteDescriptorSet textures_write{};
  textures_write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  textures_write.dstSet          = m_descriptor.shared.set;
  textures_write.dstBinding      = SharedBindings::Textures;
  textures_write.descriptorCount = (u32) textures_info.size();
  textures_write.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  textures_write.pImageInfo      = textures_info.data();

  // VkDescriptorBufferInfo sphere_descriptor{};
  // sphere_descriptor.buffer = m_spheres.gpu_data.spheres.handle;
  // sphere_descriptor.offset = 0;
  // sphere_descriptor.range  = VK_WHOLE_SIZE;

  // VkWriteDescriptorSet spheres_write{};
  // spheres_write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  // spheres_write.dstSet          = m_descriptor.shared.set;
  // spheres_write.dstBinding      = SharedBindings::Spheres;
  // spheres_write.descriptorCount = 1;
  // spheres_write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  // spheres_write.pBufferInfo     = &sphere_descriptor;

  std::array<VkWriteDescriptorSet, 6> write_descriptor_sets = //
      {
        as_write,                                             //
        image_write,                                          //
        ubo_write,                                            //
        scene_write,                                          //
        textures_write,                                       //
        primitive_write,
        // spheres_write                                         //
      };

  vkUpdateDescriptorSets(context.device(), (u32) write_descriptor_sets.size(), write_descriptor_sets.data(), 0, nullptr);
}

void RayTracer::create_pipeline() {

  Context &context = m_context_ref;

  // TODO: is there a better way to do this?
  enum stage_indices : size_t {
    generation   = 0, //
    miss         = 1, //
    close_hit    = 2, //
    sphere_hit   = 3, //
    sphere_int   = 4, //
    stages_count = 5
  };

  // SHADER STAGES
  std::array<VkPipelineShaderStageCreateInfo, (size_t) stage_indices::stages_count> stages{};

  stages[stage_indices::generation].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[stage_indices::generation].stage  = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  stages[stage_indices::generation].module = context.create_shader_module("./spv/default.rgen.spv");
  stages[stage_indices::generation].pName  = "main";

  stages[stage_indices::miss].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[stage_indices::miss].stage  = VK_SHADER_STAGE_MISS_BIT_KHR;
  stages[stage_indices::miss].module = context.create_shader_module("./spv/default.rmiss.spv");
  stages[stage_indices::miss].pName  = "main";

  stages[stage_indices::close_hit].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[stage_indices::close_hit].stage  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  stages[stage_indices::close_hit].module = context.create_shader_module("./spv/default.rchit.spv");
  stages[stage_indices::close_hit].pName  = "main";

  stages[stage_indices::sphere_hit].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[stage_indices::sphere_hit].stage  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  stages[stage_indices::sphere_hit].module = context.create_shader_module("./spv/sphere.rchit.spv");
  stages[stage_indices::sphere_hit].pName  = "main";

  stages[stage_indices::sphere_int].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[stage_indices::sphere_int].stage  = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
  stages[stage_indices::sphere_int].module = context.create_shader_module("./spv/sphere.rint.spv");
  stages[stage_indices::sphere_int].pName  = "main";

  // SHADER GROUPS
  /*
    we have three types of shader groups:
    - ray generation group (only ray_generation)
    - ray miss group (only miss)
    - ray hit group (close_hit, any_hit, intersection)

    current layout:
    /-------------\ --------
    | raygen      |  generation
    |-------------| --------
    | miss        |  miss
    |-------------| --------
    | hit         |  hit
    |-------------| --------
    | sphere hit  |  sphere's shaders
    | sphere int  |
    \-------------/ --------

    enum miss_indices : size_t {
      default_miss = 0, //
      miss_count   = 1
    };

    enum hit_indicies: size_t{
      default_hit = 0,
      hit_count = 1
    };
  */
  auto const gen_count  = 1;
  auto const miss_count = 1;
  auto const hit_count  = 1;
  m_shader_groups.resize(gen_count + miss_count + hit_count);

  // 1. ray generation group
  m_shader_groups[stage_indices::generation].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
  m_shader_groups[stage_indices::generation].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  m_shader_groups[stage_indices::generation].generalShader      = stage_indices::generation;
  m_shader_groups[stage_indices::generation].closestHitShader   = VK_SHADER_UNUSED_KHR;
  m_shader_groups[stage_indices::generation].anyHitShader       = VK_SHADER_UNUSED_KHR;
  m_shader_groups[stage_indices::generation].intersectionShader = VK_SHADER_UNUSED_KHR;

  // 2. ray miss group
  m_shader_groups[stage_indices::miss].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
  m_shader_groups[stage_indices::miss].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  m_shader_groups[stage_indices::miss].generalShader      = stage_indices::miss;
  m_shader_groups[stage_indices::miss].closestHitShader   = VK_SHADER_UNUSED_KHR;
  m_shader_groups[stage_indices::miss].anyHitShader       = VK_SHADER_UNUSED_KHR;
  m_shader_groups[stage_indices::miss].intersectionShader = VK_SHADER_UNUSED_KHR;

  // 3. ray hit group
  m_shader_groups[stage_indices::close_hit].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
  m_shader_groups[stage_indices::close_hit].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  m_shader_groups[stage_indices::close_hit].generalShader      = VK_SHADER_UNUSED_KHR;
  m_shader_groups[stage_indices::close_hit].closestHitShader   = stage_indices::close_hit;
  m_shader_groups[stage_indices::close_hit].anyHitShader       = VK_SHADER_UNUSED_KHR;
  m_shader_groups[stage_indices::close_hit].intersectionShader = VK_SHADER_UNUSED_KHR;

  // 4. sphere hit group
  // m_shader_groups[stage_indices::sphere_hit].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
  // m_shader_groups[stage_indices::sphere_hit].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
  // m_shader_groups[stage_indices::sphere_hit].generalShader      = VK_SHADER_UNUSED_KHR;
  // m_shader_groups[stage_indices::sphere_hit].closestHitShader   = stage_indices::sphere_hit;
  // m_shader_groups[stage_indices::sphere_hit].anyHitShader       = VK_SHADER_UNUSED_KHR;
  // m_shader_groups[stage_indices::sphere_hit].intersectionShader = stage_indices::sphere_int;

  // PIPELINE LAYOUT
  VkPushConstantRange pc_range = {};
  pc_range.offset              = 0;
  pc_range.size                = sizeof(push_constant_t);
  pc_range.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_CALLABLE_BIT_KHR;

  VkPipelineLayoutCreateInfo pipeline_layout_create_info{};
  pipeline_layout_create_info.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_create_info.setLayoutCount         = 1;
  pipeline_layout_create_info.pSetLayouts            = &m_descriptor.shared.layout;
  pipeline_layout_create_info.pushConstantRangeCount = 1;
  pipeline_layout_create_info.pPushConstantRanges    = &pc_range;

  check(
      vkCreatePipelineLayout(context.device(), &pipeline_layout_create_info, nullptr, &m_pipeline_layout), //
      "creating pipeline layout for raytracing pipeline"
  );

  // RAYTRACING PIPELINE
  VkRayTracingPipelineCreateInfoKHR raytracing_pipeline_create_info{};
  raytracing_pipeline_create_info.sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
  raytracing_pipeline_create_info.stageCount                   = (u32) stages.size();
  raytracing_pipeline_create_info.pStages                      = stages.data();
  raytracing_pipeline_create_info.groupCount                   = (u32) m_shader_groups.size();
  raytracing_pipeline_create_info.pGroups                      = m_shader_groups.data();
  raytracing_pipeline_create_info.maxPipelineRayRecursionDepth = 2;
  raytracing_pipeline_create_info.layout                       = m_pipeline_layout;

  check(
      vkCreateRayTracingPipelinesKHR(context.device(), VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &raytracing_pipeline_create_info, nullptr, &m_pipeline), //
      "creating raytracing pipeline"
  );

  // SHADER BINDING TABLE

  auto     handle_count = gen_count + miss_count + hit_count;
  uint32_t handle_size  = m_rt_prop.shaderGroupHandleSize;

  // The SBT (buffer) need to have starting groups to be aligned and handles in the group to be aligned.
  u32 handle_size_aligned = align_up(handle_size, m_rt_prop.shaderGroupHandleAlignment);

  m_gen_region.stride = align_up(handle_size_aligned, m_rt_prop.shaderGroupBaseAlignment);
  m_gen_region.size   = m_gen_region.stride; // The size member of pRayGenShaderBindingTable must be equal to its stride member

  m_miss_region.stride = handle_size_aligned;
  m_miss_region.size   = align_up(miss_count * handle_size_aligned, m_rt_prop.shaderGroupBaseAlignment);

  m_hit_region.stride = handle_size_aligned;
  m_hit_region.size   = align_up(hit_count * handle_size_aligned, m_rt_prop.shaderGroupBaseAlignment);

  // Get the shader group handles
  u32                  data_size = handle_count * handle_size;
  std::vector<uint8_t> handles(data_size);
  check(                    //
      vkGetRayTracingShaderGroupHandlesKHR(
          context.device(), //
          m_pipeline,       //
          0,                // first_group
          handle_count,     // group_count
          data_size,        //
          handles.data()
      )
  );

  VkDeviceSize sbt_size = m_gen_region.size + m_miss_region.size + m_hit_region.size + m_call_region.size;

  VkBufferCreateInfo sbt_buffer_info{};
  sbt_buffer_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  sbt_buffer_info.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR;
  sbt_buffer_info.size        = sbt_size;
  sbt_buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo sbt_buffer_alloc = {};
  sbt_buffer_alloc.usage                   = VMA_MEMORY_USAGE_CPU_TO_GPU;
  sbt_buffer_alloc.flags                   = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  sbt_buffer_alloc.preferredFlags          = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  check(
      vmaCreateBuffer(context.vma_allocator(), &sbt_buffer_info, &sbt_buffer_alloc, &m_sbtb_buffer.handle, &m_sbtb_buffer.allocation, nullptr), //
      "allocating buffe for Shader Binding Table"
  );
  context.set_debug_name(m_sbtb_buffer.handle, "SBT buffer");

  // Find the SBT addresses of each group
  VkDeviceAddress sbt_address = context.get_buffer_device_address(m_sbtb_buffer.handle);
  m_gen_region.deviceAddress  = sbt_address;
  m_miss_region.deviceAddress = sbt_address + m_gen_region.size;
  m_hit_region.deviceAddress  = sbt_address + m_gen_region.size + m_miss_region.size;

  void* raw_data = nullptr;
  vmaMapMemory(context.vma_allocator(), m_sbtb_buffer.allocation, &raw_data);

  // NOLINTBEGIN
  // Helper to retrieve the handle data
  auto get_handle = [&](int i) { return handles.data() + i * handle_size; };

  u8* p_sbt        = reinterpret_cast<u8*>(raw_data);
  u8* p_data       = nullptr;
  u32 handle_index = 0;

  // ray generation
  p_data = p_sbt;
  memcpy(p_data, get_handle(handle_index), handle_size);
  handle_index += 1;
  // ray miss
  p_data = p_sbt + m_gen_region.size;
  for (u32 c = 0; c < miss_count; c++) {
    memcpy(p_data, get_handle(handle_index), handle_size);
    handle_index += 1;
    p_data += m_miss_region.stride;
  }
  // hit
  p_data = p_sbt + m_gen_region.size + m_miss_region.size;
  for (u32 c = 0; c < hit_count; c++) {
    memcpy(p_data, get_handle(handle_index), handle_size);
    handle_index += 1;
    p_data += m_hit_region.stride;
  }
  // NOLINTEND
  vmaUnmapMemory(context.vma_allocator(), m_sbtb_buffer.allocation);

  for (auto stage : stages) {
    vkDestroyShaderModule(context.device(), stage.module, nullptr);
  }
}

// void RayTracer::load_spheres(std::vector<std::pair<sphere_t, u32>> &spheres, std::vector<material_options> &materials) {
//   Context &context = m_context_ref.get();
//   // LOADING TO CPU
//   m_spheres.raw.spheres.reserve(spheres.size());
//   m_spheres.raw.aabbs.reserve(spheres.size());
//   m_spheres.raw.mat_indices.reserve(spheres.size());
//   m_spheres.raw.materials.reserve(materials.size());
//   for (auto [s, index] : spheres) {
//     m_spheres.raw.spheres.push_back(s);
//     m_spheres.raw.mat_indices.push_back(index);
//     aabb_t aabb{};
//     aabb.min = s.center - glm::vec3(s.radius);
//     aabb.max = s.center + glm::vec3(s.radius);
//     m_spheres.raw.aabbs.push_back(aabb);
//   }
//   for (auto &option : materials) {
//     material m{};
//     m.base_color_factor = option.color;
//     if (not option.texture_name.empty()) {
//       m.base_color_texture = (i32) m_spheres.raw.texture_names.size();
//       m_spheres.raw.texture_names.push_back(option.texture_name);
//     }
//     m_spheres.raw.materials.push_back(m);
//   }
//   // LOADING TO GPU
//   m_spheres.gpu_data.spheres_total = (u32) m_spheres.raw.spheres.size();
//   VkBufferUsageFlags flag = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
//   m_spheres.gpu_data.spheres        = context.create_buffer(m_spheres.raw.spheres, flag | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
//   m_spheres.gpu_data.material       = context.create_buffer(m_spheres.raw.materials, flag | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
//   m_spheres.gpu_data.material_index = context.create_buffer(m_spheres.raw.mat_indices, flag | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
//   m_spheres.gpu_data.aabbs          = context.create_buffer(m_spheres.raw.aabbs, flag);
//   context.set_debug_name(m_spheres.gpu_data.spheres.handle, "spheres data");
//   context.set_debug_name(m_spheres.gpu_data.material.handle, "spheres material");
//   context.set_debug_name(m_spheres.gpu_data.material_index.handle, "spheres material_index");
//   context.set_debug_name(m_spheres.gpu_data.aabbs.handle, "aabbs");
//   mesh_description desc{};
//   desc.txt_offset             = (u32) m_textures.size();
//   desc.material_address       = context.get_buffer_device_address(m_spheres.gpu_data.material.handle);
//   desc.material_index_address = context.get_buffer_device_address(m_spheres.gpu_data.material_index.handle);
//   for (auto &text_name : m_spheres.raw.texture_names) {
//     auto file_path = fmt::format("../assets/textures/{}", text_name);
//     m_textures.push_back(create_texture(file_path).value_or(m_default_texture));
//   }
//   m_spheres.desc_index = (u32) m_description.data.size();
//   m_description.data.push_back(desc);
//   // BLAS creation
//   VkAccelerationStructureGeometryAabbsDataKHR aabbs{};
//   aabbs.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
//   aabbs.data.deviceAddress = context.get_buffer_device_address(m_spheres.gpu_data.aabbs.handle);
//   aabbs.stride             = sizeof(aabb_t);
//   // The bottom level acceleration structure
//   VkAccelerationStructureGeometryKHR acceleration_structure_geometry{};
//   acceleration_structure_geometry.sType          = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
//   acceleration_structure_geometry.geometryType   = VK_GEOMETRY_TYPE_AABBS_KHR;
//   acceleration_structure_geometry.flags          = VK_GEOMETRY_OPAQUE_BIT_KHR;
//   acceleration_structure_geometry.geometry.aabbs = aabbs;
//   // Get the size requirements for buffers involved in the acceleration structure build process
//   VkAccelerationStructureBuildGeometryInfoKHR acceleration_structure_build_geometry_info{};
//   acceleration_structure_build_geometry_info.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
//   acceleration_structure_build_geometry_info.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
//   acceleration_structure_build_geometry_info.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
//   acceleration_structure_build_geometry_info.geometryCount = 1;
//   acceleration_structure_build_geometry_info.pGeometries   = &acceleration_structure_geometry;
//   VkAccelerationStructureBuildSizesInfoKHR acceleration_structure_build_sizes_info{};
//   acceleration_structure_build_sizes_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
//   vkGetAccelerationStructureBuildSizesKHR(
//       context.device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &acceleration_structure_build_geometry_info, &m_spheres.gpu_data.spheres_total,
//       &acceleration_structure_build_sizes_info
//   );
//   VkBufferCreateInfo acc_buffer_info{};
//   acc_buffer_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
//   acc_buffer_info.usage       = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
//   acc_buffer_info.size        = acceleration_structure_build_sizes_info.accelerationStructureSize;
//   acc_buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
//   VmaAllocationCreateInfo acc_buffer_alloc{};
//   acc_buffer_alloc.usage = VMA_MEMORY_USAGE_GPU_ONLY;
//   check(
//       vmaCreateBuffer(
//           context.vma_allocator(),             //
//           &acc_buffer_info, &acc_buffer_alloc, //
//           &m_spheres.blas.buffer.handle, &m_spheres.blas.buffer.allocation, nullptr
//       ),
//       "creating buffer for spheres blas"
//   );
//   context.set_debug_name(m_spheres.blas.buffer.handle, fmt::format("spheres blas buffer"));
//   // Create the acceleration structure
//   VkAccelerationStructureCreateInfoKHR acceleration_structure_create_info{};
//   acceleration_structure_create_info.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
//   acceleration_structure_create_info.buffer = m_spheres.blas.buffer.handle;
//   acceleration_structure_create_info.size   = acceleration_structure_build_sizes_info.accelerationStructureSize;
//   acceleration_structure_create_info.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
//   check(vkCreateAccelerationStructureKHR(context.device(), &acceleration_structure_create_info, nullptr, &m_spheres.blas.handle));
//   // The actual build process starts here
//   buffer_t           scratch_buffer = {};
//   VkBufferCreateInfo scratch_buffer_info{};
//   scratch_buffer_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
//   scratch_buffer_info.usage       = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
//   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT; scratch_buffer_info.size        = acceleration_structure_build_sizes_info.buildScratchSize;
//   scratch_buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
//   VmaAllocationCreateInfo scratch_buffer_alloc = {};
//   scratch_buffer_alloc.usage                   = VMA_MEMORY_USAGE_CPU_TO_GPU;
//   check(
//       vmaCreateBuffer(
//           context.vma_allocator(),                     //
//           &scratch_buffer_info, &scratch_buffer_alloc, //
//           &scratch_buffer.handle, &scratch_buffer.allocation, nullptr
//       ),
//       "creating scratch buffer for spheres blas"
//   );
//   VkAccelerationStructureBuildGeometryInfoKHR acceleration_build_geometry_info{};
//   acceleration_build_geometry_info.sType                     = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
//   acceleration_build_geometry_info.type                      = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
//   acceleration_build_geometry_info.flags                     = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
//   acceleration_build_geometry_info.mode                      = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
//   acceleration_build_geometry_info.dstAccelerationStructure  = m_spheres.blas.handle;
//   acceleration_build_geometry_info.geometryCount             = 1;
//   acceleration_build_geometry_info.pGeometries               = &acceleration_structure_geometry;
//   acceleration_build_geometry_info.scratchData.deviceAddress = context.get_buffer_device_address(scratch_buffer.handle);
//   VkAccelerationStructureBuildRangeInfoKHR acceleration_structure_build_range_info{};
//   acceleration_structure_build_range_info.firstVertex     = 0;
//   acceleration_structure_build_range_info.primitiveCount  = (u32) m_spheres.gpu_data.spheres_total;
//   acceleration_structure_build_range_info.primitiveOffset = 0;
//   acceleration_structure_build_range_info.transformOffset = 0;
//   std::array<VkAccelerationStructureBuildRangeInfoKHR*, 1> acceleration_build_structure_range_infos = { &acceleration_structure_build_range_info };
//   context.immediate_submit([&](VkCommandBuffer cmd) {
//     vkCmdBuildAccelerationStructuresKHR(cmd, 1, &acceleration_build_geometry_info, acceleration_build_structure_range_infos.data());
//   });
//   vmaDestroyBuffer(context.vma_allocator(), scratch_buffer.handle, scratch_buffer.allocation);
//   glm::mat3x4          rtxT             = glm::transpose(glm::mat4{ 1.f });
//   VkTransformMatrixKHR transform_matrix = {};
//   memcpy(&transform_matrix, glm::value_ptr(rtxT), sizeof(VkTransformMatrixKHR));
//   auto blas = m_spheres.blas.handle;
//   VkAccelerationStructureDeviceAddressInfoKHR acceleration_device_address_info{};
//   acceleration_device_address_info.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
//   acceleration_device_address_info.accelerationStructure = blas;
//   auto device_address = vkGetAccelerationStructureDeviceAddressKHR(context.device(), &acceleration_device_address_info);
//   VkAccelerationStructureInstanceKHR blas_instance{};
//   blas_instance.transform                              = transform_matrix;
//   blas_instance.instanceCustomIndex                    = m_spheres.desc_index;
//   blas_instance.mask                                   = 0xFF;
//   blas_instance.instanceShaderBindingTableRecordOffset = 1;
//   blas_instance.flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
//   blas_instance.accelerationStructureReference         = device_address;
//   m_blas_instances.push_back(blas_instance);
//   // Instance creation
// }

void RayTracer::create_tlas() {
  Context &context = m_context_ref;

  buffer_t instances =
      context.create_buffer(m_blas_instances, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  VkDeviceAddress instance_address = context.get_buffer_device_address(instances.handle);

  context.set_debug_name(instances.handle, "tlas instances buffer");

  // The top level acceleration structure contains (bottom level) instance as the input geometry
  VkAccelerationStructureGeometryKHR acceleration_structure_geometry{};
  acceleration_structure_geometry.sType                                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  acceleration_structure_geometry.geometryType                          = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  acceleration_structure_geometry.flags                                 = VK_GEOMETRY_OPAQUE_BIT_KHR;
  acceleration_structure_geometry.geometry.instances.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  acceleration_structure_geometry.geometry.instances.arrayOfPointers    = VK_FALSE;
  acceleration_structure_geometry.geometry.instances.data.deviceAddress = instance_address;

  // Get the size requirements for buffers involved in the acceleration structure build process
  VkAccelerationStructureBuildGeometryInfoKHR acceleration_structure_build_geometry_info{};
  acceleration_structure_build_geometry_info.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  acceleration_structure_build_geometry_info.type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  acceleration_structure_build_geometry_info.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  acceleration_structure_build_geometry_info.geometryCount = 1;
  acceleration_structure_build_geometry_info.pGeometries   = &acceleration_structure_geometry;

  const auto primitive_count = static_cast<u32>(m_blas_instances.size());

  VkAccelerationStructureBuildSizesInfoKHR acceleration_structure_build_sizes_info{};
  acceleration_structure_build_sizes_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  vkGetAccelerationStructureBuildSizesKHR(
      context.device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &acceleration_structure_build_geometry_info, &primitive_count,
      &acceleration_structure_build_sizes_info
  );

  // Create a buffer to hold the acceleration structure
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
          &m_tlas.buffer.handle, &m_tlas.buffer.allocation, nullptr
      ),
      "creating buffer for tlas"
  );

  // Create the acceleration structure
  VkAccelerationStructureCreateInfoKHR acceleration_structure_create_info{};
  acceleration_structure_create_info.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
  acceleration_structure_create_info.buffer = m_tlas.buffer.handle;
  acceleration_structure_create_info.size   = acceleration_structure_build_sizes_info.accelerationStructureSize;
  acceleration_structure_create_info.type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  check(
      vkCreateAccelerationStructureKHR(context.device(), &acceleration_structure_create_info, nullptr, &m_tlas.handle), //
      "creating tlas"
  );

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
      "creating scratch buffer for tlas"
  );

  VkAccelerationStructureBuildGeometryInfoKHR acceleration_build_geometry_info{};
  acceleration_build_geometry_info.sType                     = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  acceleration_build_geometry_info.type                      = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  acceleration_build_geometry_info.flags                     = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  acceleration_build_geometry_info.mode                      = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  acceleration_build_geometry_info.dstAccelerationStructure  = m_tlas.handle;
  acceleration_build_geometry_info.geometryCount             = 1;
  acceleration_build_geometry_info.pGeometries               = &acceleration_structure_geometry;
  acceleration_build_geometry_info.scratchData.deviceAddress = context.get_buffer_device_address(scratch_buffer.handle);

  VkAccelerationStructureBuildRangeInfoKHR acceleration_structure_build_range_info{};
  acceleration_structure_build_range_info.primitiveCount                                            = primitive_count;
  acceleration_structure_build_range_info.primitiveOffset                                           = 0;
  acceleration_structure_build_range_info.firstVertex                                               = 0;
  acceleration_structure_build_range_info.transformOffset                                           = 0;
  std::array<VkAccelerationStructureBuildRangeInfoKHR*, 1> acceleration_build_structure_range_infos = { &acceleration_structure_build_range_info };

  // Build the acceleration structure on the device via a one-time command buffer submission
  context.immediate_submit([&](VkCommandBuffer cmd) {
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &acceleration_build_geometry_info, acceleration_build_structure_range_infos.data());
  });

  vmaDestroyBuffer(context.vma_allocator(), instances.handle, instances.allocation);
  vmaDestroyBuffer(context.vma_allocator(), scratch_buffer.handle, scratch_buffer.allocation);
}

texture_t RayTracer::create_texture(
    u32 width, u32 height,            //
    std::vector<unsigned char> &data, //
    VkFilter mag_filter, VkFilter min_filter, VkFormat format
) {
  Context &context = m_context_ref;

  texture_t result = {};
  result.width     = width;
  result.height    = height;
  result.format    = format;

  VkImageCreateInfo image_create_info{};
  image_create_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_create_info.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  image_create_info.imageType     = VK_IMAGE_TYPE_2D;
  image_create_info.format        = result.format;
  image_create_info.extent.width  = result.width;
  image_create_info.extent.height = result.height;
  image_create_info.extent.depth  = 1;
  image_create_info.mipLevels     = mip_levels(result.width, result.height);
  image_create_info.arrayLayers   = 1;
  image_create_info.samples       = VK_SAMPLE_COUNT_1_BIT;
  image_create_info.tiling        = VK_IMAGE_TILING_OPTIMAL;
  image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  auto* raw_data = data.data();
  result.image   = context.create_image_on_gpu(image_create_info, raw_data, data.size());

  context.generate_mipmaps(result.image.handle, image_create_info);

  VkImageViewCreateInfo image_view_create_info{};
  image_view_create_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  image_view_create_info.image                           = result.image.handle;
  image_view_create_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
  image_view_create_info.format                          = image_create_info.format;
  image_view_create_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  image_view_create_info.subresourceRange.baseMipLevel   = 0;
  image_view_create_info.subresourceRange.levelCount     = image_create_info.mipLevels;
  image_view_create_info.subresourceRange.baseArrayLayer = 0;
  image_view_create_info.subresourceRange.layerCount     = 1;

  check(
      vkCreateImageView(context.device(), &image_view_create_info, nullptr, &result.view), //
      "creating view for texture"
  );

  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(context.physical_device(), &properties);

  VkSamplerCreateInfo sampler_create_info{};
  sampler_create_info.sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sampler_create_info.magFilter               = mag_filter;
  sampler_create_info.minFilter               = min_filter;
  sampler_create_info.addressModeU            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_create_info.addressModeV            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_create_info.addressModeW            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_create_info.anisotropyEnable        = VK_TRUE;
  sampler_create_info.maxAnisotropy           = properties.limits.maxSamplerAnisotropy;
  sampler_create_info.borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  sampler_create_info.unnormalizedCoordinates = VK_FALSE;
  sampler_create_info.compareEnable           = VK_FALSE;
  sampler_create_info.compareOp               = VK_COMPARE_OP_ALWAYS;
  sampler_create_info.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  sampler_create_info.maxLod                  = static_cast<float>(image_create_info.mipLevels);
  // sampler_create_info.minLod     = sampler_create_info.maxLod - 1.f;
  sampler_create_info.minLod     = 0;
  sampler_create_info.mipLodBias = 0.0f;

  check(
      vkCreateSampler(context.device(), &sampler_create_info, nullptr, &result.sampler), //
      "creating sampler for texture"
  );

  return result;
}

void RayTracer::reset_frame() { m_shader_frame = 0; }

} // namespace whim::vk