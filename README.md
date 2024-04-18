# Дипломный проект: "Разработка системы рендеринга на основе метода трассировки лучей"


## DEPENDENCIES

all dependencies are stored in `external/` folder
- [fmt](https://github.com/fmtlib/fmt) 
- [GLFW](https://github.com/glfw/glfw)
- [GLM](https://github.com/g-truc/glm)
- [tinyglfw](https://github.com/syoyo/tinygltf)
- [vk-bootstrap](https://github.com/charles-lunarg/vk-bootstrap)
- [VulkanMemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)


## How to run

if you want to build this code you need to install VulkanSDK in your system

```bash
cd build 
cmake .. # add your options
cmake --build .
cd ../bin
./main.exe
```

## ROADMAP

- [x] Window creation
- [x] Vulkan initialization
- [x] Simple input system
- [x] ImGUI support
- [x] GLTF scene loading
  - [x] Accelerated Structure Creation  
  - [x] Textures loading and creation
- [x] Raytracing Pipeline creation  
  - [x] Shader Binding Table Creating 
- [ ] Procedural primitives support
  - [ ] Spheres
- [ ] Main shaders
  - [x] triangle hit group
  - [ ] shadow miss shader 
  - [ ] procedural hit group
  - [ ] any hit shaders
- [ ] PBR
  - [ ] light sources
  - [ ] BSDF implementation
  - [ ] path tracing 
  - [ ] transparent objects 