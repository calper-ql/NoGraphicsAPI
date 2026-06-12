# ngapi — the embeddable NoGraphicsAPI library

This folder is the entire reusable library, packaged so you can drop it into
your own CMake project in one step. Everything outside it (samples, windowing
backends, tests) is development scaffolding for NGAPI itself and is not needed
to use the API.

## Requirements

- CMake 3.22+
- A C++20 compiler
- Vulkan headers + loader (anything `find_package(Vulkan)` can find, e.g. the
  [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) or your distro's
  `vulkan-headers` / `vulkan-icd-loader` packages)

No shader compiler is needed to build or use the library — its one internal
compute shader ships pre-compiled, embedded in `src/PatchDescriptorsSpv.h`.
You only need `slangc` (or any compiler producing SPIR-V) for your own
shaders.

## Using it

Pick whichever fits your project; all of them end the same way:

```cmake
target_link_libraries(your_app PRIVATE ngapi::ngapi)
```

**1. Whole repo as a git submodule (or clone):**

```sh
git submodule add <repo-url> external/NoGraphicsAPI
```
```cmake
add_subdirectory(external/NoGraphicsAPI)
```

Samples and tests are skipped automatically when the repo is not the top-level
project, so only the library (and its one dependency) builds.

**2. Just this folder, copied into your tree:**

```cmake
add_subdirectory(path/to/ngapi)
```

**3. No subdirectory at all — plain include:**

```cmake
include(path/to/ngapi/ngapi.cmake)
```

**4. FetchContent:**

```cmake
include(FetchContent)
FetchContent_Declare(NoGraphicsAPI
    GIT_REPOSITORY <repo-url>
    GIT_TAG        main
    GIT_SUBMODULES "")   # submodules are samples-only; vk-bootstrap is fetched on demand
FetchContent_MakeAvailable(NoGraphicsAPI)
```

Then in your code:

```c++
#include "NoGraphicsAPI.h"
```

See the top-level [README](../README.md) for an API walkthrough.

## The vk-bootstrap dependency

NGAPI uses [vk-bootstrap](https://github.com/charles-lunarg/vk-bootstrap) for
Vulkan instance/device setup, linked privately (it does not appear in NGAPI's
headers). `ngapi.cmake` resolves it automatically, in this order:

1. a `vk-bootstrap::vk-bootstrap` target your project already defines
2. `-DNGAPI_VK_BOOTSTRAP_DIR=<path>` — a local checkout you point it at
3. the repo's `external/vk-bootstrap` submodule, when this folder lives in the
   full NoGraphicsAPI repo and the submodule is initialized
4. FetchContent — downloaded at configure time, pinned to the same tag as the
   submodule

So with the full repo either initialize the submodule or let FetchContent
grab it; with a copied `ngapi/` folder it just downloads (or use 1/2 for
offline builds).

## Compiling your own shaders

`NoGraphicsAPI.h` is shared between C++ and slang — pass ngapi's include
directory (exported as `NGAPI_INCLUDE_DIR`) to `slangc -I` when compiling your
shader code. `cmake/CompileShaders.cmake` is a ready-made helper for that;
`ngapi.cmake` adds it to `CMAKE_MODULE_PATH`:

```cmake
include(CompileShaders)
set(NGAPI_SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")
compile_shader(SOURCE shaders/MyShader.slang STAGE compute OUTPUT MyShader.spv)
add_custom_target(my_shaders ALL DEPENDS ${NGAPI_SHADER_OUTPUTS})
```

`slangc` is picked up from `PATH` or the `VULKAN_SDK` environment variable;
pass `-DSLANGC=/path/to/slangc` to override.

## Shared CPU/GPU struct layout rules

Structs passed to shaders via device-buffer pointers must have identical layout
on the CPU (C++) and on every GPU target. The two compilers disagree in
opposite directions, so both rules below are needed:

1. **Never use `float3` (or `uint3`/`int3`) as a field in a device-buffer
   struct.** Metal MSL gives 3-component vectors 16-byte alignment inside
   structs, effectively padding them to 16 bytes, while the C++ `float3` is
   12 bytes — every field that follows sits at a different offset on each
   side. Use `float4` with `w = 0.0` and access `.rgb` / `.xyz` in the shader.
   (Bare `float3*` pointers to tightly packed arrays are fine: the Metal
   shader toolchain rewrites those to `packed_float3*`.)

2. **Explicitly pad the struct to a multiple of 16 bytes if it is read as an
   array.** Slang's SPIR-V "natural" layout packs fields tightly and does NOT
   round the array stride up to the largest member's alignment:
   `{ float4; float }` has stride 20 in SPIR-V but 32 in MSL and in C++ with
   `alignas(16)`. Add explicit `float _pad...;` members so the packed field
   sum equals the C++ `sizeof` — then all three layouts agree. For the same
   reason, keep every `float4` field at an offset that is a multiple of 16.

Use `NGAPI_ASSERT_GPU_STRUCT` to pin the agreed size at compile time (defined
in `NoGraphicsAPI_Impl.h` for C++, a no-op when the header is compiled as
shader code):

```cpp
struct alignas(16) MyData
{
    float4 color;                // rule 1: NOT float3
    float  value;
    float  _pad0, _pad1, _pad2;  // rule 2: explicit tail padding to 32
};
NGAPI_ASSERT_GPU_STRUCT(MyData, 32);
```

The assert fires immediately if the C++ size deviates from the declared one,
making layout drift a compile error rather than a silent corruption. The
compute sample's `Tint` struct (`samples/compute/Compute.h`) is the worked
example: with `float3 color` it broke Metal (stride 16 vs 32), with an
unpadded `float4` it broke Vulkan (stride 32 vs 20) — the committed form is
the one whose stride all targets agree on.

## Developing NGAPI itself

`src/PatchDescriptorsSpv.h` is generated from `shaders/PatchDescriptors.slang`
and checked in. After editing that shader, regenerate it (run these from the
repository root, one level above this folder) — preferably with the slangc
pinned in `ci/Dockerfile` so the header stays reproducible:

```sh
docker build -t ngapi-ci ci   # once
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/src -w /src ngapi-ci \
    cmake -P ngapi/cmake/EmbedPatchDescriptors.cmake
```

or, with your own slangc on PATH, `cmake -P ngapi/cmake/EmbedPatchDescriptors.cmake`
(also available as the `ngapi-embed-patch-descriptors` build target in the full
repo).
