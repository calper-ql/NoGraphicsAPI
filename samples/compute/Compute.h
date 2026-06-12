#ifndef SAMPLES_SHADER_COMPUTE_H
#define SAMPLES_SHADER_COMPUTE_H

#include "NoGraphicsAPI.h"

// A plain data struct kept in its own GPU buffer. ComputeData below holds a
// pointer to an array of these; the shader follows that pointer to read them.
// Arrayed device-buffer structs must agree on stride across C++, SPIR-V and
// MSL — float4 (not float3) for vectors, explicit tail padding to a 16-byte
// multiple. See "Shared CPU/GPU struct layout rules" in ngapi/README.md.
struct alignas(16) Tint
{
    float4 color;    // xyz = RGB tint, w unused
    float  strength;
    float  _pad0, _pad1, _pad2;
};
NGAPI_ASSERT_GPU_STRUCT(Tint, 32);

struct alignas(16) ComputeData
{
    uint2 imageSize;
    uint srcTexture;
    uint dstTexture;

    // A regular buffer attached by pointer: the shader dereferences `tints`
    // directly (a GPU address), no descriptor binding involved.
    Tint* tints;
    uint tintCount;
};

#endif // SAMPLES_SHADER_COMPUTE_H