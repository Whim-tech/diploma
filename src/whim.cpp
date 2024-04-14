
#define VMA_IMPLEMENTATION
#include <vma/vk_mem_alloc.h>

#define TINYOBJLOADER_IMPLEMENTATION
// #define TINYOBJLOADER_USE_MAPBOX_EARCUT gives robust trinagulation. Requires C++11
#include "tiny_obj_loader.h"


#define TINYGLTF_NOEXCEPTION 
#define TINYGLTF_NO_STB_IMAGE_WRITE 
#define TINYGLTF_USE_CPP14 
#define TINYGLTF_IMPLEMENTATION
#include "tiny_gltf.h"

#define STB_IMAGE_IMPLEMENTATION
#include "external/stb_image.h"