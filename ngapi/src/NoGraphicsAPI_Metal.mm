// Metal backend — covers gpuCreateDevice, gpuMalloc, gpuDispatch, textures,
// samplers, argument buffers, surface/swapchain (CAMetalLayer), and blit.
// Runs the headless compute samples (multiplegpus, learning) and the windowed
// compute sample (CAMetalLayer → ngapi::createWindow via window_metal.mm).
//
// Friction log (things that diverge from the Vulkan backend):
//
//   1. Encoder switching: Metal splits work into typed encoders (blit, compute,
//      render). Switching type requires ending the current encoder. The Vulkan
//      backend has no such concept — barriers and copies sit in the same stream.
//      We track which encoder is live and end it lazily on type change.
//
//   2. gpuDispatch data pointer: Vulkan passes a raw VkDeviceAddress via push
//      constants; the shader dereferences it directly. Slang's MSL output wraps
//      the entry-point parameters in `EntryPointParams_N constant* [[buffer(N)]]`
//      where the struct contains the data pointer. We use setBytes(&gpuAddr,8,N)
//      to pass the 8-byte GPU address as a constant buffer — the shader reads
//      `entryPointParams->data` and dereferences it as a device pointer (works
//      on Apple Silicon's unified address space). N is determined at pipeline
//      creation via MTLComputePipelineReflection (single-entry .metal files
//      always get N=0; multi-entry files assign N by entry-point order).
//
//   3. Shader format: Vulkan loads SPIR-V. Metal compiles from MSL source
//      (.metal files produced by `slangc -target metal`) at pipeline creation
//      time. gpuCreateComputePipeline detects the MTLB magic header for pre-
//      compiled .metallib, and falls back to runtime MSL compilation otherwise.
//
//   4. Timeline semaphores: mapped 1:1 onto MTLSharedEvent, which supports
//      signaling at a value and notifying a listener at a threshold — identical
//      semantics to VkSemaphoreTypeTimeline.
//
//   5. MEMORY_GPU: Private storage mode has no CPU-visible pointer. We return
//      the MTLBuffer's GPU address cast to void* (same convention as the Vulkan
//      backend's MEMORY_GPU path, which returns the VkDeviceAddress).
//
//   6. Samplers: Slang carries INLINE_SAMPLER state in tagged 32-bit literals
//      (`ngapiSamplerHeap[0x4Exxxxxx]`) and STATIC_SAMPLER state in the default
//      of a 64-bit function constant (0x4E47 in the top 16 bits). We scan the
//      MSL source at pipeline creation, decode each unique state into an
//      MTLSamplerState, assign compact slot indices starting at 1 (slot 0 is
//      the device default sampler, matching the Vulkan backend's heap), patch
//      inline literals to their slot, and bind a sampler argument buffer at
//      dispatch time. STATIC_SAMPLER constants are resolved by SPECIALIZING
//      the function (newFunctionWithName:constantValues:error:) — building any
//      pipeline from a plain-fetched function out of a library that declares
//      function constants aborts the process (uncatchable validateWithDevice
//      assertion, with or without MTL_DEBUG_LAYER), so the constantValues path
//      is mandatory, not an optimization. Metal samplers have no LOD bias;
//      a non-zero STATIC_SAMPLER lodBias is logged and ignored.
//
//   7. Bindless texture/sampler heaps: Slang emits `textureHeap[]`,
//      `rwTextureHeap[]`, and `ngapiSamplerHeap[]` as unbounded kernel
//      parameters. These become Metal Tier-2 argument buffers. The CPU-side
//      GpuTextureDescriptor stores the id<MTLTexture> raw pointer; gpuDispatch
//      builds argument buffers from it on every call.
//
//   8. Threadgroup size: Slang does NOT emit `[[max_total_threads_per_threadgroup]]`
//      in Metal output. The build system embeds "// NGAPI_THREADS X Y Z" at the
//      top of each .metal file; gpuCreateComputePipeline reads it and stores the
//      size in GpuPipeline_T. Defaults to {64,1,1} when the comment is absent.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <QuartzCore/CAMetalLayer.h>
#import <dispatch/dispatch.h>

#define GPU_EXPOSE_INTERNAL
#include "NoGraphicsAPI.h"

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <cassert>

// ---------------------------------------------------------------------------
// Internal structs
// ---------------------------------------------------------------------------

struct MetalAllocation
{
    id<MTLBuffer> buffer;
    void*    cpuPtr;   // nil for MEMORY_GPU
    uint64_t gpuAddr;
    size_t   size;
};

struct MetalDevice
{
    id<MTLDevice> device;

    // Lazily-compiled compute PSO used by gpuBlitTexture.
    id<MTLComputePipelineState> blitPso = nil;

    std::mutex             allocMutex;
    std::vector<MetalAllocation> allocations;

    // Raw (unretained) id<MTLTexture> pointers of live gpuCreateTexture
    // textures. gpuDispatch validates texture-heap descriptors against this
    // registry: the heap pointer is often a suballocation of a larger
    // gpuMalloc page whose tail holds unrelated user data, and the heap scan
    // must not mistake that data for descriptors. Guarded by allocMutex.
    std::vector<const void*> textures;

    // allocMutex must be held.
    bool isLiveTexture(const void* p) const
    {
        return std::find(textures.begin(), textures.end(), p) != textures.end();
    }

    MetalAllocation* findByCpu(void* ptr)
    {
        auto* p = reinterpret_cast<uint8_t*>(ptr);
        for (auto& a : allocations)
            if (a.cpuPtr && p >= reinterpret_cast<uint8_t*>(a.cpuPtr)
                         && p <  reinterpret_cast<uint8_t*>(a.cpuPtr) + a.size)
                return &a;
        return nullptr;
    }

    MetalAllocation* findByGpu(uint64_t addr)
    {
        for (auto& a : allocations)
            if (addr >= a.gpuAddr && addr < a.gpuAddr + a.size)
                return &a;
        return nullptr;
    }
};

struct GpuDevice_T          { MetalDevice* metalDevice = nullptr; };
struct GpuTexture_T         { id<MTLTexture> texture; GpuDevice device; };

// Swapchain image is a value embedded in GpuSwapchain_T; gpuSwapchainImage
// returns a stable pointer to it — valid until the next call or gpuDestroySwapchain.
struct GpuSwapchain_T
{
    CAMetalLayer*     layer;
    id<MTLDrawable>   currentDrawable = nil;
    GpuTexture_T      swapchainTex{};    // reused each frame
    GpuDevice         device;
    uint32_t          framesInFlight;
};

struct GpuPipeline_T
{
    id<MTLComputePipelineState> pso;
    id<MTLFunction>             fn;               // retained for argument encoder creation
    GpuDevice                   device;
    NSUInteger                  entryParamsIndex   = 0;
    uint3                       threadgroupSize    = {64, 1, 1};

    // Buffer indices in the kernel for each resource array (NSUIntegerMax = absent)
    NSUInteger textureHeapIndex   = NSUIntegerMax;
    NSUInteger rwTextureHeapIndex = NSUIntegerMax;
    NSUInteger samplerHeapIndex   = NSUIntegerMax;

    // Sampler argument buffer — built once at pipeline creation from the
    // sampler declarations found in the MSL source. The pipeline owns the
    // sampler objects: the argument buffer only stores their GPU handles,
    // which dangle if the MTLSamplerStates are released.
    id<MTLBuffer> samplerArgBuf = nil;
    std::vector<id<MTLSamplerState>> samplers;
};

struct GpuQueue_T             { id<MTLCommandQueue>         queue; GpuDevice device; };
struct GpuSemaphore_T         { id<MTLSharedEvent>          event; GpuDevice device; };
struct GpuDepthStencilState_T { GpuDepthStencilDesc desc; };
struct GpuBlendState_T        { GpuBlendDesc desc; };
#ifdef GPU_RAY_TRACING_EXTENSION
struct GpuAccelerationStructure_T { void* placeholder; };
#endif

struct GpuCommandBuffer_T
{
    id<MTLCommandBuffer>          cb;
    id<MTLBlitCommandEncoder>     blitEnc    = nil;
    id<MTLComputeCommandEncoder>  computeEnc = nil;
    GpuPipeline currentPipeline = nullptr;
    GpuDevice   device;
    void*       activeTextureHeapGpu = nullptr; // set by gpuSetActiveTextureHeapPtr

    id<MTLBlitCommandEncoder> blit()
    {
        if (computeEnc) { [computeEnc endEncoding]; computeEnc = nil; }
        if (!blitEnc)    blitEnc = [cb blitCommandEncoder];
        return blitEnc;
    }

    id<MTLComputeCommandEncoder> compute()
    {
        if (blitEnc)    { [blitEnc endEncoding];    blitEnc    = nil; }
        if (!computeEnc) computeEnc = [cb computeCommandEncoder];
        return computeEnc;
    }

    void endAll()
    {
        if (blitEnc)    { [blitEnc endEncoding];    blitEnc    = nil; }
        if (computeEnc) { [computeEnc endEncoding]; computeEnc = nil; }
    }
};

#ifdef GPU_SURFACE_EXTENSION
struct GpuSurface_T   { CAMetalLayer* layer; };
// GpuSwapchain_T is defined above (before GpuPipeline_T) so gpuSwapchainImage
// can return &swapchain->swapchainTex without a forward reference issue.
#endif

// ---------------------------------------------------------------------------
// Global device list (populated once on gpuCreateInstance)
// ---------------------------------------------------------------------------

static std::vector<GpuDeviceDesc> g_deviceDescs;
static std::vector<id<MTLDevice>> g_devices;

static void populateDevices()
{
    if (!g_devices.empty()) return;
    for (id<MTLDevice> d in MTLCopyAllDevices())
    {
        GpuDeviceDesc desc = {};
        strncpy(desc.name, [[d name] UTF8String], sizeof(desc.name) - 1);
        desc.dedicatedMemory = [d recommendedMaxWorkingSetSize];
        desc.discrete        = ![d hasUnifiedMemory];
        g_deviceDescs.push_back(desc);
        g_devices.push_back(d);
    }
}

// ---------------------------------------------------------------------------
// Instance
// ---------------------------------------------------------------------------

RESULT gpuCreateInstance()
{
    populateDevices();
    return g_devices.empty() ? RESULT_FAILURE : RESULT_SUCCESS;
}

void gpuDestroyInstance()
{
    g_devices.clear();
    g_deviceDescs.clear();
}

// Vulkan-specific handle accessors — return nullptr in the Metal backend.
#ifdef GPU_EXPOSE_INTERNAL
void* gpuVulkanInstance() { return nullptr; }

// metalLayer is a CAMetalLayer* passed from window_metal.mm::createSurface().
GpuSurface gpuCreateSurface(void* metalLayer)
{
    return new GpuSurface_T{ (__bridge CAMetalLayer*)metalLayer };
}

void* gpuVulkanSurface(GpuSurface surface)
{
    return (__bridge void*)surface->layer;
}

void gpuDestroySurface(GpuSurface surface) { delete surface; }
#endif

// ---------------------------------------------------------------------------
// Device enumeration
// ---------------------------------------------------------------------------

uint32_t gpuDeviceCount()
{
    populateDevices();
    return static_cast<uint32_t>(g_devices.size());
}

GpuDeviceDesc gpuDeviceDesc(uint32_t index)
{
    populateDevices();
    if (index >= g_deviceDescs.size()) return {};
    return g_deviceDescs[index];
}

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

GpuDevice gpuCreateDevice(uint32_t deviceIndex)
{
    populateDevices();
    if (deviceIndex >= g_devices.size()) return nullptr;

    auto* md    = new MetalDevice();
    md->device  = g_devices[deviceIndex];
    return new GpuDevice_T{ md };
}

void gpuDestroyDevice(GpuDevice device)
{
    if (!device) return;
    delete device->metalDevice;
    delete device;
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

static void* mallocImpl(MetalDevice* md, size_t bytes, size_t align, MEMORY memory)
{
    bytes = (bytes + align - 1) & ~(align - 1);

    MTLResourceOptions opts;
    bool cpuVisible;
    switch (memory)
    {
    case MEMORY_DEFAULT:
        opts       = MTLResourceStorageModeShared;
        cpuVisible = true;
        break;
    case MEMORY_GPU:
        opts       = MTLResourceStorageModePrivate;
        cpuVisible = false;
        break;
    case MEMORY_READBACK:
        opts       = MTLResourceStorageModeShared;
        cpuVisible = true;
        break;
    }

    id<MTLBuffer> buf = [md->device newBufferWithLength:bytes options:opts];

    MetalAllocation a;
    a.buffer  = buf;
    a.cpuPtr  = cpuVisible ? [buf contents] : nullptr;
    a.gpuAddr = [buf gpuAddress];
    a.size    = bytes;

    {
        std::lock_guard lock(md->allocMutex);
        md->allocations.push_back(a);
    }

    return cpuVisible ? [buf contents] : reinterpret_cast<void*>(a.gpuAddr);
}

void* gpuMalloc(GpuDevice device, size_t bytes, MEMORY memory)
{
    return mallocImpl(device->metalDevice, bytes, GPU_DEFAULT_ALIGNMENT, memory);
}

void* gpuMalloc(GpuDevice device, size_t bytes, size_t align, MEMORY memory)
{
    return mallocImpl(device->metalDevice, bytes, align, memory);
}

void gpuFree(GpuDevice device, void* ptr)
{
    auto* md = device->metalDevice;
    std::lock_guard lock(md->allocMutex);
    auto it = std::find_if(md->allocations.begin(), md->allocations.end(),
        [ptr](const MetalAllocation& a) {
            return a.cpuPtr == ptr || reinterpret_cast<void*>(a.gpuAddr) == ptr;
        });
    if (it != md->allocations.end())
        md->allocations.erase(it);
}

void* gpuHostToDevicePointer(GpuDevice device, void* ptr)
{
    auto* md = device->metalDevice;
    std::lock_guard lock(md->allocMutex);
    MetalAllocation* a = md->findByCpu(ptr);
    if (!a) return ptr;
    size_t offset = reinterpret_cast<uint8_t*>(ptr) - reinterpret_cast<uint8_t*>(a->cpuPtr);
    return reinterpret_cast<void*>(a->gpuAddr + offset);
}

// ---------------------------------------------------------------------------
// Format / texture helpers
// ---------------------------------------------------------------------------

static MTLPixelFormat metalPixelFormat(FORMAT fmt)
{
    switch (fmt)
    {
    case FORMAT_RGBA8_UNORM:    return MTLPixelFormatRGBA8Unorm;
    case FORMAT_BGRA8_SRGB:     return MTLPixelFormatBGRA8Unorm_sRGB;
    case FORMAT_D32_FLOAT:      return MTLPixelFormatDepth32Float;
    case FORMAT_RG11B10_FLOAT:  return MTLPixelFormatRG11B10Float;
    case FORMAT_RGB10_A2_UNORM: return MTLPixelFormatRGB10A2Unorm;
    case FORMAT_RG32_FLOAT:     return MTLPixelFormatRG32Float;
    case FORMAT_RGBA32_FLOAT:   return MTLPixelFormatRGBA32Float;
    case FORMAT_RGBA16_FLOAT:   return MTLPixelFormatRGBA16Float;
    case FORMAT_RGB32_FLOAT:    return MTLPixelFormatInvalid; // no packed RGB32 in Metal
    default:                    return MTLPixelFormatInvalid;
    }
}

static MTLTextureType metalTextureType(TEXTURE type)
{
    switch (type)
    {
    case TEXTURE_1D:         return MTLTextureType1D;
    case TEXTURE_2D:         return MTLTextureType2D;
    case TEXTURE_3D:         return MTLTextureType3D;
    case TEXTURE_CUBE:       return MTLTextureTypeCube;
    case TEXTURE_2D_ARRAY:   return MTLTextureType2DArray;
    case TEXTURE_CUBE_ARRAY: return MTLTextureTypeCubeArray;
    default:                 return MTLTextureType2D;
    }
}

static MTLTextureUsage metalTextureUsage(USAGE_FLAGS flags)
{
    MTLTextureUsage usage = MTLTextureUsageUnknown;
    if (flags & USAGE_SAMPLED)                  usage |= MTLTextureUsageShaderRead;
    if (flags & USAGE_STORAGE)                  usage |= MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    if (flags & USAGE_COLOR_ATTACHMENT)         usage |= MTLTextureUsageRenderTarget;
    if (flags & USAGE_DEPTH_STENCIL_ATTACHMENT) usage |= MTLTextureUsageRenderTarget;
    return usage;
}

static NSUInteger metalBytesPerPixel(MTLPixelFormat fmt)
{
    switch (fmt)
    {
    case MTLPixelFormatRGBA8Unorm:
    case MTLPixelFormatBGRA8Unorm_sRGB:
    case MTLPixelFormatRGB10A2Unorm:
    case MTLPixelFormatRG11B10Float:
    case MTLPixelFormatDepth32Float:
        return 4;
    case MTLPixelFormatRGBA16Float:
    case MTLPixelFormatRG32Float:
        return 8;
    case MTLPixelFormatRGBA32Float:
        return 16;
    default:
        return 4;
    }
}

// ---------------------------------------------------------------------------
// Sampler helpers
// ---------------------------------------------------------------------------

static MTLSamplerAddressMode metalAddressMode(uint32_t mode)
{
    switch (mode)
    {
    case 0:  return MTLSamplerAddressModeRepeat;
    case 1:  return MTLSamplerAddressModeClampToEdge;
    case 2:  return MTLSamplerAddressModeMirrorRepeat;
    case 3:  return MTLSamplerAddressModeClampToBorderColor;
    case 4:  return MTLSamplerAddressModeMirrorClampToEdge;
    default: return MTLSamplerAddressModeRepeat;
    }
}

static MTLSamplerBorderColor metalBorderColor(uint32_t b)
{
    switch (b)
    {
    case 1:  return MTLSamplerBorderColorOpaqueBlack;
    case 2:  return MTLSamplerBorderColorOpaqueWhite;
    default: return MTLSamplerBorderColorTransparentBlack;
    }
}

static MTLCompareFunction metalCompareFunction(uint32_t op)
{
    switch (op)
    {
    case 1:  return MTLCompareFunctionNever;
    case 2:  return MTLCompareFunctionLess;
    case 3:  return MTLCompareFunctionEqual;
    case 4:  return MTLCompareFunctionLessEqual;
    case 5:  return MTLCompareFunctionGreater;
    case 6:  return MTLCompareFunctionNotEqual;
    case 7:  return MTLCompareFunctionGreaterEqual;
    case 8:  return MTLCompareFunctionAlways;
    default: return MTLCompareFunctionNever; // COMPARE_NONE → not a shadow sampler
    }
}

// Decode an NGAPI packed 48-bit sampler state into an MTLSamplerState.
// INLINE_SAMPLER 32-bit literals are widened to this form before the call
// (lodBias 0 encoded as 128, minLod 0, maxLod unbounded).
//
// Bit layout (from Sampler.h NGAPI_SAMPLER_BITS + STATIC_SAMPLER):
//   bit  0     minFilter        bits 17-18  borderColor
//   bit  1     magFilter        bits 19-22  compareOp (0 = none, 1..8 = VK-style ops)
//   bit  2     mipFilter        bits 23-30  lodBias, (bias+8)*16 — no Metal equivalent
//   bits 3-5   addressU         bits 31-38  minLod, lod*16
//   bits 6-8   addressV         bits 39-46  maxLod, lod*16 (255 = unbounded)
//   bits 9-11  addressW
//   bits 12-16 maxAnisotropy
static id<MTLSamplerState> createSamplerFromState(id<MTLDevice> device, uint64_t state)
{
    MTLSamplerDescriptor* desc = [MTLSamplerDescriptor new];
    desc.minFilter       = (state & 0x01) ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    desc.magFilter       = (state & 0x02) ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    desc.mipFilter       = (state & 0x04) ? MTLSamplerMipFilterLinear   : MTLSamplerMipFilterNearest;
    desc.sAddressMode    = metalAddressMode((uint32_t)(state >>  3) & 0x7);
    desc.tAddressMode    = metalAddressMode((uint32_t)(state >>  6) & 0x7);
    desc.rAddressMode    = metalAddressMode((uint32_t)(state >>  9) & 0x7);
    uint32_t aniso       = (uint32_t)(state >> 12) & 0x1F;
    desc.maxAnisotropy   = (aniso == 0) ? 1 : aniso;
    desc.borderColor     = metalBorderColor((uint32_t)(state >> 17) & 0x3);
    desc.compareFunction = metalCompareFunction((uint32_t)(state >> 19) & 0xF);

    // Metal samplers have no LOD bias; every shader in the repo declares 0.0.
    const float lodBias = (float)((state >> 23) & 255) / 16.0f - 8.0f;
    if (lodBias != 0.0f)
        NSLog(@"NoGraphicsAPI Metal: sampler lodBias %.3f ignored (MTLSamplerDescriptor has no LOD bias)", lodBias);

    desc.lodMinClamp = (float)((state >> 31) & 255) / 16.0f;
    const uint32_t maxLodBits = (uint32_t)(state >> 39) & 255;
    desc.lodMaxClamp = (maxLodBits == 255) ? FLT_MAX : (float)maxLodBits / 16.0f;

    // Required for samplers written through an MTLArgumentEncoder.
    desc.supportArgumentBuffers = YES;
    return [device newSamplerStateWithDescriptor:desc];
}

// Packed state of the device default sampler at heap slot 0: min/mag LINEAR,
// mip NEAREST, repeat xyz, aniso off, opaque-black border, no compare,
// lodBias 0 (encoded 128), minLod 0, maxLod 0 — the exact zero-initialized
// VkSamplerCreateInfo the Vulkan backend builds in gpuCreateDevice (maxLod 0
// clamps to the base mip; it is NOT unbounded).
static constexpr uint64_t kDefaultSamplerState = 0x3ull | (1ull << 17) | (128ull << 23);

// Result of scanning one MSL source for sampler declarations. Slot 0 is
// always the device default sampler — the Vulkan backend reserves heap slot 0
// the same way, and shaders may read `ngapiSamplerHeap[0]` raw (samplerbench's
// benchHardware does).
struct SamplerScanResult
{
    std::vector<id<MTLSamplerState>> samplers;                      // heap slot -> sampler
    std::vector<std::pair<NSUInteger, uint64_t>> functionConstants; // [[function_constant(i)]] -> value to specialize
};

// Scan MSL source for both Slang sampler-declaration forms:
//
//   INLINE_SAMPLER  — tagged 32-bit literals (top byte 0x4E): decoded, deduped
//                     and patched in-place to the allocated heap slot.
//   STATIC_SAMPLER  — slangc lowers the 64-bit spec constant to
//                       constant ulong fc_X [[function_constant(N)]];
//                       constant ulong X = is_function_constant_defined(fc_X)
//                                              ? fc_X : <0x4E47-magic>ULL;
//                     The packed state lives in the default literal; the heap
//                     slot is injected by specializing constant N at function
//                     creation (out.functionConstants). The source is left
//                     untouched — once the constant is defined, the ternary
//                     folds to it and the magic default is dead code.
//
// Both forms dedup on the canonical 48-bit state (inline literals widened with
// neutral LOD fields exactly like the Vulkan backend), so identical effective
// states share one hardware sampler. Returns the patched MSL string.
static std::string patchSamplerLiterals(
    const std::string& src,
    id<MTLDevice>      device,
    SamplerScanResult& out)
{
    out.samplers.push_back(createSamplerFromState(device, kDefaultSamplerState));

    // Canonical 48-bit state -> heap slot (slots start at 1; slot 0 is the
    // default sampler and is not part of the dedup map, matching Vulkan).
    std::vector<std::pair<uint64_t, uint32_t>> seen;
    auto slotFor = [&](uint64_t state) -> uint32_t {
        for (const auto& e : seen)
            if (e.first == state) return e.second;
        uint32_t slot = (uint32_t)out.samplers.size();
        seen.push_back({state, slot});
        out.samplers.push_back(createSamplerFromState(device, state));
        return slot;
    };

    // --- STATIC_SAMPLER: function-constant declarations -------------------
    static const char kFcAttr[] = "[[function_constant(";
    const size_t kFcAttrLen = sizeof(kFcAttr) - 1;
    for (size_t at = src.find(kFcAttr); at != std::string::npos; at = src.find(kFcAttr, at + 1))
    {
        // Constant index inside the attribute.
        size_t d = at + kFcAttrLen;
        bool haveIndex = d < src.size() && isdigit((unsigned char)src[d]);
        NSUInteger index = 0;
        while (d < src.size() && isdigit((unsigned char)src[d]))
            index = index * 10 + (NSUInteger)(src[d++] - '0');

        // Declared name: walk back over whitespace, then identifier chars.
        size_t end = at;
        while (end > 0 && isspace((unsigned char)src[end - 1])) --end;
        size_t begin = end;
        while (begin > 0 && (isalnum((unsigned char)src[begin - 1]) || src[begin - 1] == '_')) --begin;
        std::string name = src.substr(begin, end - begin);
        if (!haveIndex || name.empty())
            continue;

        // Paired default: the first `<name> : <digits>ULL` after the
        // declaration (the false arm of the is_function_constant_defined
        // ternary slangc emits).
        uint64_t value = 0;
        bool haveValue = false;
        for (size_t use = src.find(name, at + 1); use != std::string::npos; use = src.find(name, use + 1))
        {
            size_t p = use + name.size();
            while (p < src.size() && isspace((unsigned char)src[p])) ++p;
            if (p >= src.size() || src[p] != ':') continue;
            ++p;
            while (p < src.size() && isspace((unsigned char)src[p])) ++p;
            if (p >= src.size() || !isdigit((unsigned char)src[p])) continue;
            uint64_t v = 0;
            while (p < src.size() && isdigit((unsigned char)src[p]))
                v = v * 10 + (uint64_t)(src[p++] - '0');
            value = v;
            haveValue = true;
            break;
        }
        if (!haveValue)
        {
            // Every function constant MUST be specialized (see friction log 6);
            // leaving it unset makes function creation fail with an NSError,
            // which gpuCreateComputePipeline reports.
            assert(false && "Metal: [[function_constant]] without a parsable integer default");
            continue;
        }

        if ((value & STATIC_SAMPLER_MAGIC_MASK) == STATIC_SAMPLER_MAGIC)
            out.functionConstants.push_back({index, (uint64_t)slotFor(value & ~STATIC_SAMPLER_MAGIC_MASK)});
        else
            // Not a sampler constant (nothing in the repo emits these today):
            // specialize it to its own default, which preserves semantics.
            out.functionConstants.push_back({index, value});
    }

    // --- INLINE_SAMPLER: tagged 32-bit literals, patched in place ---------
    // Inline sampler magic: top byte == 0x4E (see INLINE_SAMPLER_MAGIC in Sampler.h)
    static constexpr uint32_t MAGIC_MASK = 0xFF000000u;
    static constexpr uint32_t MAGIC_VAL  = 0x4E000000u;

    std::string result;
    result.reserve(src.size());

    size_t i = 0;
    while (i < src.size())
    {
        if (!isdigit((unsigned char)src[i]))
        {
            result += src[i++];
            continue;
        }

        // Scan a run of digits
        size_t j = i;
        while (j < src.size() && isdigit((unsigned char)src[j])) ++j;

        bool hasU = (j < src.size()) && (src[j] == 'U' || src[j] == 'u');
        if (!hasU)
        {
            result.append(src, i, j - i);
            i = j;
            continue;
        }

        // Parse the number; cap at uint32 to avoid overflow (64-bit ULL
        // literals — including STATIC_SAMPLER defaults — are left verbatim,
        // they are handled by the function-constant pass above).
        uint64_t val64 = 0;
        bool overflow = false;
        for (size_t k = i; k < j; ++k)
        {
            val64 = val64 * 10 + (unsigned char)(src[k] - '0');
            if (val64 > 0xFFFFFFFFull) { overflow = true; break; }
        }

        if (!overflow && ((uint32_t)val64 & MAGIC_MASK) == MAGIC_VAL)
        {
            // Canonicalize to the 48-bit layout exactly like the Vulkan
            // backend: bits 0-22 of the literal, lodBias 0 (encoded 128),
            // minLod 0, maxLod unbounded.
            const uint64_t state = (val64 & 0x7fffff) | (128ull << 23) | (255ull << 39);
            result += std::to_string(slotFor(state));
            result += src[j]; // 'U' or 'u'
            i = j + 1;
            continue;
        }

        // Not a sampler literal — copy verbatim including the U suffix
        result.append(src, i, (j - i) + 1);
        i = j + 1;
    }

    return result;
}

// Find the name of the function carrying `attr` ("[[kernel]]", "[[vertex]]",
// "[[fragment]]") in MSL source by locating the '(' that opens its parameter
// list and walking back over the identifier. library.functionNames exposes
// only entry points and slangc renames entries ('main' -> 'main_0'; render
// libraries contain non-entry helpers with entry-like names), so the source
// is the only reliable name authority. When `entry` is non-null, prefers the
// attributed function whose name matches or starts with it; falls back to the
// first attributed function (per-entry .metal files contain exactly one).
// Returns "" when no attributed function exists.
static std::string findEntryName(const std::string& src, const char* attr, const char* entry)
{
    std::string fallback;
    const size_t attrLen = strlen(attr);
    for (size_t at = src.find(attr); at != std::string::npos; at = src.find(attr, at + 1))
    {
        size_t paren = src.find('(', at + attrLen);
        if (paren == std::string::npos) continue;
        size_t end = paren;
        while (end > at && isspace((unsigned char)src[end - 1])) --end;
        size_t begin = end;
        while (begin > at && (isalnum((unsigned char)src[begin - 1]) || src[begin - 1] == '_')) --begin;
        std::string name = src.substr(begin, end - begin);
        if (name.empty()) continue;
        if (entry && name.compare(0, strlen(entry), entry) == 0) return name;
        if (fallback.empty()) fallback = name;
    }
    return fallback;
}

// ---------------------------------------------------------------------------
// Textures
// ---------------------------------------------------------------------------

GpuTextureSizeAlign gpuTextureSizeAlign(GpuDevice, GpuTextureDesc)
{
    // Metal creates textures as standalone objects; the caller's gpuMalloc
    // allocation (texturePtr) is a token we ignore at texture creation time.
    // Return a minimal size so gpuMalloc doesn't allocate a zero-byte buffer.
    return { GPU_DEFAULT_ALIGNMENT, GPU_DEFAULT_ALIGNMENT };
}

GpuTexture gpuCreateTexture(GpuDevice device, GpuTextureDesc desc, void* /*ptrGpu*/)
{
    MTLTextureDescriptor* td = [MTLTextureDescriptor new];
    td.textureType      = metalTextureType(desc.type);
    td.pixelFormat      = metalPixelFormat(desc.format);
    td.width            = desc.dimensions.x;
    td.height           = (desc.type == TEXTURE_1D) ? 1 : desc.dimensions.y;
    td.depth            = (desc.type == TEXTURE_3D) ? desc.dimensions.z : 1;
    td.mipmapLevelCount = desc.mipCount;
    td.arrayLength      = (desc.type == TEXTURE_2D_ARRAY || desc.type == TEXTURE_CUBE_ARRAY)
                              ? desc.layerCount : 1;
    td.sampleCount      = desc.sampleCount;
    td.usage            = metalTextureUsage(desc.usage);
    td.storageMode      = MTLStorageModePrivate;

    id<MTLTexture> tex = [device->metalDevice->device newTextureWithDescriptor:td];
    if (!tex)
    {
        NSLog(@"gpuCreateTexture: newTextureWithDescriptor failed");
        return nullptr;
    }
    {
        std::lock_guard lock(device->metalDevice->allocMutex);
        device->metalDevice->textures.push_back((__bridge const void*)tex);
    }
    return new GpuTexture_T{ tex, device };
}

void gpuDestroyTexture(GpuTexture texture)
{
    if (!texture) return;
    auto* md = texture->device->metalDevice;
    {
        std::lock_guard lock(md->allocMutex);
        auto& v = md->textures;
        v.erase(std::remove(v.begin(), v.end(), (__bridge const void*)texture->texture), v.end());
    }
    delete texture;
}

// Pack the id<MTLTexture> raw pointer into GpuTextureDescriptor so gpuDispatch
// can reconstruct it. data[1]=0 → sampled (textureHeap[]), data[1]=1 → storage
// (rwTextureHeap[]). The caller must keep the owning GpuTexture alive.
GpuTextureDescriptor gpuTextureViewDescriptor(GpuTexture texture, GpuViewDesc)
{
    GpuTextureDescriptor d = {};
    d.data[0] = (uint64_t)(__bridge void*)texture->texture;
    d.data[1] = 0; // sampled
    return d;
}

GpuTextureDescriptor gpuRWTextureViewDescriptor(GpuTexture texture, GpuViewDesc)
{
    GpuTextureDescriptor d = {};
    d.data[0] = (uint64_t)(__bridge void*)texture->texture;
    d.data[1] = 1; // storage / read-write
    return d;
}

// ---------------------------------------------------------------------------
// Pipelines
// ---------------------------------------------------------------------------

GpuPipeline gpuCreateComputePipeline(GpuDevice device, ByteSpan ir, const char* entry)
{
    auto* md = device->metalDevice;
    NSError* err = nil;
    id<MTLLibrary> lib = nil;

    // Detect compiled .metallib by its "MTLB" magic (4D 54 4C 42).
    bool isMetallib = ir.size() >= 4
        && ir[0] == 0x4D && ir[1] == 0x54 && ir[2] == 0x4C && ir[3] == 0x42;

    // --- Threadgroup size: read "// NGAPI_THREADS X Y Z" from first line ---
    uint3 tgSize = {64, 1, 1};
    if (!isMetallib && ir.size() > 20)
    {
        const char* p = reinterpret_cast<const char*>(ir.data());
        if (strncmp(p, "// NGAPI_THREADS ", 17) == 0)
        {
            unsigned x = 64, y = 1, z = 1;
            sscanf(p + 17, "%u %u %u", &x, &y, &z);
            tgSize = {x, y, z};
        }
    }

    SamplerScanResult samplerScan; // MSL source path only; samplers[0] = default
    std::string patched;           // patched MSL source (kept for the scans below)

    if (isMetallib)
    {
        dispatch_data_t data = dispatch_data_create(
            ir.data(), ir.size(), nil, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        lib = [md->device newLibraryWithData:data error:&err];
    }
    else
    {
        // Treat bytes as UTF-8 MSL source. Patch inline sampler magic literals
        // before compilation so the ngapiSamplerHeap index stays small, and
        // collect the STATIC_SAMPLER function-constant slots.
        std::string src(reinterpret_cast<const char*>(ir.data()), ir.size());
        patched = patchSamplerLiterals(src, md->device, samplerScan);

        NSString* msl = [NSString stringWithUTF8String:patched.c_str()];
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.languageVersion = MTLLanguageVersion3_1;
        lib = [md->device newLibraryWithSource:msl options:opts error:&err];
    }

    if (!lib)
    {
        NSLog(@"gpuCreateComputePipeline: library error: %@", err);
        return nullptr;
    }

    // Entry-name resolution. For MSL source, scan for the [[kernel]] function
    // (see findEntryName — functionNames/hardcoded fallbacks are unreliable);
    // for .metallib there is no source, keep the historical 'main_0' mapping.
    NSString* entryName = nil;
    if (isMetallib)
    {
        entryName = entry ? [NSString stringWithUTF8String:entry] : @"main_0";
        if ([entryName isEqualToString:@"main"]) entryName = @"main_0";
    }
    else
    {
        std::string scanned = findEntryName(patched, "[[kernel]]", entry);
        if (scanned.empty())
        {
            NSLog(@"gpuCreateComputePipeline: no [[kernel]] function in MSL source (entry '%s')",
                  entry ? entry : "(default)");
            return nullptr;
        }
        entryName = [NSString stringWithUTF8String:scanned.c_str()];
    }

    // A library whose source declares function constants MUST have its
    // functions created via newFunctionWithName:constantValues:error: with
    // every constant set — building any pipeline state from a plain-fetched
    // function out of such a library aborts the process (uncatchable Metal
    // validateWithDevice assertion; there is no NSError fallback).
    id<MTLFunction> fn = nil;
    if (!isMetallib && patched.find("[[function_constant(") != std::string::npos)
    {
        MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
        for (const auto& fc : samplerScan.functionConstants)
        {
            uint64_t value = fc.second;
            [cv setConstantValue:&value type:MTLDataTypeULong atIndex:fc.first];
        }
        fn = [lib newFunctionWithName:entryName constantValues:cv error:&err];
    }
    else
    {
        fn = [lib newFunctionWithName:entryName];
    }
    if (!fn)
    {
        NSLog(@"gpuCreateComputePipeline: entry '%@' not found or specialization failed: %@",
              entryName, err);
        return nullptr;
    }

    MTLComputePipelineReflection* refl = nil;
    id<MTLComputePipelineState> pso =
        [md->device newComputePipelineStateWithFunction:fn
                                               options:MTLPipelineOptionBindingInfo
                                            reflection:&refl
                                                 error:&err];
    if (!pso)
    {
        NSLog(@"gpuCreateComputePipeline: pipeline error: %@", err);
        return nullptr;
    }

    // Discover binding indices from reflection.
    NSUInteger paramsIndex    = 0;
    NSUInteger texHeapIdx     = NSUIntegerMax;
    NSUInteger rwTexHeapIdx   = NSUIntegerMax;
    NSUInteger samplerHeapIdx = NSUIntegerMax;
    for (id<MTLBinding> b in refl.bindings)
    {
        if (b.type != MTLBindingTypeBuffer) continue;
        if ([b.name hasPrefix:@"rwTextureHeap"])          rwTexHeapIdx   = b.index;
        else if ([b.name hasPrefix:@"textureHeap"])        texHeapIdx     = b.index;
        else if ([b.name hasPrefix:@"ngapiSamplerHeap"])   samplerHeapIdx = b.index;
        else if ([b.name hasPrefix:@"entryPointParams"])   paramsIndex    = b.index;
    }

    // Pre-build sampler argument buffer (stays constant for this pipeline).
    // The samplers vector is never empty on the MSL source path (slot 0 is the
    // default sampler), so any kernel that binds ngapiSamplerHeap gets a real
    // heap — including ones with zero patched literals that read slot 0 raw.
    id<MTLBuffer> samplerArgBuf = nil;
    const auto& samplers = samplerScan.samplers;
    if (!samplers.empty() && samplerHeapIdx != NSUIntegerMax)
    {
        MTLArgumentDescriptor* ad = [MTLArgumentDescriptor argumentDescriptor];
        ad.dataType    = MTLDataTypeSampler;
        ad.index       = 0;
        ad.arrayLength = samplers.size();

        id<MTLArgumentEncoder> enc = [md->device newArgumentEncoderWithArguments:@[ad]];
        samplerArgBuf = [md->device newBufferWithLength:enc.encodedLength
                                               options:MTLResourceStorageModeShared];
        [enc setArgumentBuffer:samplerArgBuf offset:0];
        for (NSUInteger i = 0; i < (NSUInteger)samplers.size(); ++i)
            [enc setSamplerState:samplers[i] atIndex:i];
    }

    auto* p              = new GpuPipeline_T{};
    p->pso               = pso;
    p->fn                = fn;
    p->device            = device;
    p->entryParamsIndex  = paramsIndex;
    p->threadgroupSize   = tgSize;
    p->textureHeapIndex  = texHeapIdx;
    p->rwTextureHeapIndex= rwTexHeapIdx;
    p->samplerHeapIndex  = samplerHeapIdx;
    p->samplerArgBuf     = samplerArgBuf;
    p->samplers          = std::move(samplerScan.samplers);
    return p;
}

GpuPipeline gpuCreateGraphicsPipeline(GpuDevice, ByteSpan, ByteSpan, GpuRasterDesc)
{
    assert(false && "Metal: graphics pipeline not implemented in spike");
    return nullptr;
}

GpuPipeline gpuCreateGraphicsMeshletPipeline(GpuDevice, ByteSpan, ByteSpan, GpuRasterDesc)
{
    assert(false && "Metal: meshlet pipeline not implemented in spike");
    return nullptr;
}

void gpuFreePipeline(GpuPipeline pipeline) { delete pipeline; }

// ---------------------------------------------------------------------------
// State objects (stubs)
// ---------------------------------------------------------------------------

GpuDepthStencilState gpuCreateDepthStencilState(GpuDepthStencilDesc desc)
{
    return new GpuDepthStencilState_T{ desc };
}

GpuBlendState gpuCreateBlendState(GpuBlendDesc desc)
{
    return new GpuBlendState_T{ desc };
}

void gpuFreeDepthStencilState(GpuDepthStencilState s) { delete s; }
void gpuFreeBlendState(GpuBlendState s)               { delete s; }

// ---------------------------------------------------------------------------
// Queue
// ---------------------------------------------------------------------------

GpuQueue gpuCreateQueue(GpuDevice device)
{
    id<MTLCommandQueue> q = [device->metalDevice->device newCommandQueue];
    return new GpuQueue_T{ q, device };
}

void gpuDestroyQueue(GpuQueue queue) { delete queue; }

// ---------------------------------------------------------------------------
// Command buffer
// ---------------------------------------------------------------------------

GpuCommandBuffer gpuStartCommandRecording(GpuQueue queue)
{
    id<MTLCommandBuffer> cb = [queue->queue commandBuffer];
    auto* gcb = new GpuCommandBuffer_T{};
    gcb->cb     = cb;
    gcb->device = queue->device;
    return gcb;
}

void gpuSetPipeline(GpuCommandBuffer cb, GpuPipeline pipeline)
{
    cb->currentPipeline = pipeline;
    auto enc = cb->compute();
    [enc setComputePipelineState:pipeline->pso];
}

void gpuSetActiveTextureHeapPtr(GpuCommandBuffer cb, void* ptrGpu)
{
    cb->activeTextureHeapGpu = ptrGpu;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------
//
// Slang's MSL output wraps entry-point parameters in
//   `device EntryPointParams_N* entryPointParams_N [[buffer(N)]]`
// (after the address-space patch applied at build time). We pass the 8-byte
// GPU address of the user struct and let the shader dereference it.
//
// For textured compute shaders, we also build Tier-2 argument buffers for
// textureHeap[], rwTextureHeap[], and ngapiSamplerHeap[] and bind them at
// the buffer indices discovered via reflection at pipeline creation.

void gpuDispatch(GpuCommandBuffer cb, void* dataGpu, uint3 gridDimensions)
{
    auto* md  = cb->device->metalDevice;
    auto  enc = cb->compute();
    auto* pl  = cb->currentPipeline;

    // Bind the entry-point params (GPU address of the user data struct).
    uint64_t addr = reinterpret_cast<uint64_t>(dataGpu);
    id<MTLBuffer> paramBuf = [md->device newBufferWithBytes:&addr
                                                     length:sizeof(addr)
                                                    options:MTLResourceStorageModeShared];
    [enc setBuffer:paramBuf offset:0 atIndex:pl->entryParamsIndex];

    // Bind texture / sampler argument buffers when the heap pointer has been set
    // and the pipeline uses resource arrays.
    bool needsTexHeap = cb->activeTextureHeapGpu &&
        (pl->textureHeapIndex   != NSUIntegerMax ||
         pl->rwTextureHeapIndex != NSUIntegerMax ||
         pl->samplerHeapIndex   != NSUIntegerMax);

    if (needsTexHeap)
    {
        uint64_t heapGpuAddr = reinterpret_cast<uint64_t>(cb->activeTextureHeapGpu);

        // Find the CPU-accessible allocation that backs the texture heap.
        MetalAllocation* heapAlloc = nullptr;
        {
            std::lock_guard lock(md->allocMutex);
            heapAlloc = md->findByGpu(heapGpuAddr);
        }

        if (heapAlloc && heapAlloc->cpuPtr)
        {
            size_t offset = (size_t)(heapGpuAddr - heapAlloc->gpuAddr);
            const auto* descs = reinterpret_cast<const GpuTextureDescriptor*>(
                static_cast<const uint8_t*>(heapAlloc->cpuPtr) + offset);
            NSUInteger numDescs = (NSUInteger)((heapAlloc->size - offset) / sizeof(GpuTextureDescriptor));

            // The lock guards the texture registry; the heap may be a
            // suballocation of a larger gpuMalloc page whose tail holds
            // unrelated user data, so a slot only counts as occupied when
            // data[0] is a live gpuCreateTexture texture.
            std::lock_guard lock(md->allocMutex);
            auto occupied = [&](NSUInteger k) {
                return descs[k].data[0] && md->isLiveTexture((const void*)descs[k].data[0]);
            };

            // Scan descriptors to find the highest occupied slot for each heap type.
            NSUInteger maxSampledSlot = 0, maxStorageSlot = 0;
            for (NSUInteger k = 0; k < numDescs; ++k)
            {
                if (!occupied(k)) continue;
                if (descs[k].data[1] == 0) maxSampledSlot = k + 1;
                else                       maxStorageSlot = k + 1;
            }

            // Build and bind the sampled-texture argument buffer.
            if (pl->textureHeapIndex != NSUIntegerMax && maxSampledSlot > 0)
            {
                MTLArgumentDescriptor* ad = [MTLArgumentDescriptor argumentDescriptor];
                ad.dataType    = MTLDataTypeTexture;
                ad.textureType = MTLTextureType2D;
                ad.access      = MTLBindingAccessReadOnly;
                ad.index       = 0;
                ad.arrayLength = maxSampledSlot;

                id<MTLArgumentEncoder> argEnc = [md->device newArgumentEncoderWithArguments:@[ad]];
                id<MTLBuffer> argBuf = [md->device newBufferWithLength:argEnc.encodedLength
                                                               options:MTLResourceStorageModeShared];
                [argEnc setArgumentBuffer:argBuf offset:0];
                for (NSUInteger k = 0; k < maxSampledSlot; ++k)
                {
                    if (occupied(k) && descs[k].data[1] == 0)
                    {
                        auto tex = (__bridge id<MTLTexture>)(void*)descs[k].data[0];
                        [argEnc setTexture:tex atIndex:k];
                        [enc useResource:tex usage:MTLResourceUsageRead];
                    }
                }
                [enc setBuffer:argBuf offset:0 atIndex:pl->textureHeapIndex];
                [enc useResource:argBuf usage:MTLResourceUsageRead];
            }

            // Build and bind the storage-texture argument buffer.
            if (pl->rwTextureHeapIndex != NSUIntegerMax && maxStorageSlot > 0)
            {
                MTLArgumentDescriptor* ad = [MTLArgumentDescriptor argumentDescriptor];
                ad.dataType    = MTLDataTypeTexture;
                ad.textureType = MTLTextureType2D;
                ad.access      = MTLBindingAccessReadWrite;
                ad.index       = 0;
                ad.arrayLength = maxStorageSlot;

                id<MTLArgumentEncoder> argEnc = [md->device newArgumentEncoderWithArguments:@[ad]];
                id<MTLBuffer> argBuf = [md->device newBufferWithLength:argEnc.encodedLength
                                                               options:MTLResourceStorageModeShared];
                [argEnc setArgumentBuffer:argBuf offset:0];
                for (NSUInteger k = 0; k < maxStorageSlot; ++k)
                {
                    if (occupied(k) && descs[k].data[1] == 1)
                    {
                        auto tex = (__bridge id<MTLTexture>)(void*)descs[k].data[0];
                        [argEnc setTexture:tex atIndex:k];
                        [enc useResource:tex usage:MTLResourceUsageRead | MTLResourceUsageWrite];
                    }
                }
                [enc setBuffer:argBuf offset:0 atIndex:pl->rwTextureHeapIndex];
                [enc useResource:argBuf usage:MTLResourceUsageRead];
            }

            // Bind the pre-built sampler argument buffer.
            if (pl->samplerHeapIndex != NSUIntegerMax && pl->samplerArgBuf)
            {
                [enc setBuffer:pl->samplerArgBuf offset:0 atIndex:pl->samplerHeapIndex];
                [enc useResource:pl->samplerArgBuf usage:MTLResourceUsageRead];
            }
        }
    }

    // Declare all live GPU allocations so the GPU can access them via device
    // pointer indirection. (An MTLHeap would be more efficient.)
    {
        std::lock_guard lock(md->allocMutex);
        for (auto& a : md->allocations)
            [enc useResource:a.buffer usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    }
    [enc useResource:paramBuf usage:MTLResourceUsageRead];

    MTLSize grid        = MTLSizeMake(gridDimensions.x, gridDimensions.y, gridDimensions.z);
    MTLSize threadgroup = MTLSizeMake(pl->threadgroupSize.x, pl->threadgroupSize.y, pl->threadgroupSize.z);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
}

void gpuDispatchIndirect(GpuCommandBuffer cb, void* dataGpu, void* gridDimensionsGpu)
{
    auto* md  = cb->device->metalDevice;
    auto  enc = cb->compute();

    uint64_t dataAddr = reinterpret_cast<uint64_t>(dataGpu);
    uint64_t gridAddr = reinterpret_cast<uint64_t>(gridDimensionsGpu);

    id<MTLBuffer> paramBuf = [md->device newBufferWithBytes:&dataAddr
                                                     length:sizeof(dataAddr)
                                                    options:MTLResourceStorageModeShared];
    [enc setBuffer:paramBuf offset:0 atIndex:cb->currentPipeline->entryParamsIndex];

    {
        std::lock_guard lock(md->allocMutex);
        MetalAllocation* ga = md->findByGpu(gridAddr);
        if (ga)
            [enc dispatchThreadgroupsWithIndirectBuffer:ga->buffer
                                  indirectBufferOffset:(NSUInteger)(gridAddr - ga->gpuAddr)
                                 threadsPerThreadgroup:MTLSizeMake(
                                     cb->currentPipeline->threadgroupSize.x,
                                     cb->currentPipeline->threadgroupSize.y,
                                     cb->currentPipeline->threadgroupSize.z)];
    }
}

// ---------------------------------------------------------------------------
// Submit
// ---------------------------------------------------------------------------

void gpuSubmit(GpuQueue /*queue*/, Span<GpuCommandBuffer> commandBuffers,
               GpuSemaphore semaphore, uint64_t value)
{
    for (auto gcb : commandBuffers)
    {
        gcb->endAll();
        [gcb->cb encodeSignalEvent:semaphore->event value:value];
        [gcb->cb commit];
        delete gcb;
    }
}

// ---------------------------------------------------------------------------
// Semaphores
// ---------------------------------------------------------------------------

GpuSemaphore gpuCreateSemaphore(GpuDevice device, uint64_t initValue)
{
    id<MTLSharedEvent> ev = [device->metalDevice->device newSharedEvent];
    ev.signaledValue = initValue;
    return new GpuSemaphore_T{ ev, device };
}

void gpuWaitSemaphore(GpuSemaphore sema, uint64_t value, uint64_t /*timeout*/)
{
    if (sema->event.signaledValue >= value) return;

    dispatch_semaphore_t dsema = dispatch_semaphore_create(0);
    auto* listener = [[MTLSharedEventListener alloc] init];
    [sema->event notifyListener:listener
                        atValue:value
                          block:^(id<MTLSharedEvent>, uint64_t) {
        dispatch_semaphore_signal(dsema);
    }];
    dispatch_semaphore_wait(dsema, DISPATCH_TIME_FOREVER);
}

void gpuDestroySemaphore(GpuSemaphore sema) { delete sema; }

// ---------------------------------------------------------------------------
// Commands — blit / barrier / texture copy
// ---------------------------------------------------------------------------

void gpuMemCpy(GpuCommandBuffer cb, void* destGpu, void* srcGpu, uint64_t size)
{
    auto* md = cb->device->metalDevice;
    uint64_t srcAddr  = reinterpret_cast<uint64_t>(srcGpu);
    uint64_t dstAddr  = reinterpret_cast<uint64_t>(destGpu);

    std::lock_guard lock(md->allocMutex);
    MetalAllocation* src = md->findByGpu(srcAddr);
    MetalAllocation* dst = md->findByGpu(dstAddr);
    if (!src || !dst) { NSLog(@"gpuMemCpy: allocation not found"); return; }

    auto enc = cb->blit();
    [enc copyFromBuffer:src->buffer
           sourceOffset:(NSUInteger)(srcAddr - src->gpuAddr)
               toBuffer:dst->buffer
      destinationOffset:(NSUInteger)(dstAddr - dst->gpuAddr)
                   size:(NSUInteger)size];
}

void gpuBarrier(GpuCommandBuffer cb, STAGE /*before*/, STAGE /*after*/, HAZARD_FLAGS /*hazards*/)
{
    // Metal's memory model for buffers on Apple Silicon is coherent within a
    // command buffer; explicit barriers between blit and compute encoders are
    // implied by ending one encoder before starting the next.
    (void)cb;
}

void gpuCopyToTexture(GpuCommandBuffer cb, void* srcGpu, GpuTexture texture)
{
    auto* md = cb->device->metalDevice;
    uint64_t srcAddr = reinterpret_cast<uint64_t>(srcGpu);

    MetalAllocation* src = nullptr;
    {
        std::lock_guard lock(md->allocMutex);
        src = md->findByGpu(srcAddr);
    }
    if (!src) { NSLog(@"gpuCopyToTexture: src buffer not found"); return; }

    id<MTLTexture> tex     = texture->texture;
    NSUInteger     w       = tex.width;
    NSUInteger     h       = tex.height;
    NSUInteger     bpr     = w * metalBytesPerPixel(tex.pixelFormat);
    NSUInteger     srcOff  = (NSUInteger)(srcAddr - src->gpuAddr);

    auto enc = cb->blit();
    [enc copyFromBuffer:src->buffer
          sourceOffset:srcOff
     sourceBytesPerRow:bpr
   sourceBytesPerImage:bpr * h
            sourceSize:MTLSizeMake(w, h, 1)
             toTexture:tex
      destinationSlice:0
      destinationLevel:0
     destinationOrigin:MTLOriginMake(0, 0, 0)];
}

void gpuCopyFromTexture(GpuCommandBuffer cb, void* dstGpu, GpuTexture texture)
{
    auto* md = cb->device->metalDevice;
    uint64_t dstAddr = reinterpret_cast<uint64_t>(dstGpu);

    MetalAllocation* dst = nullptr;
    {
        std::lock_guard lock(md->allocMutex);
        dst = md->findByGpu(dstAddr);
    }
    if (!dst) { NSLog(@"gpuCopyFromTexture: dst buffer not found"); return; }

    id<MTLTexture> tex    = texture->texture;
    NSUInteger     w      = tex.width;
    NSUInteger     h      = tex.height;
    NSUInteger     bpr    = w * metalBytesPerPixel(tex.pixelFormat);
    NSUInteger     dstOff = (NSUInteger)(dstAddr - dst->gpuAddr);

    auto enc = cb->blit();
    [enc copyFromTexture:tex
            sourceSlice:0
            sourceLevel:0
           sourceOrigin:MTLOriginMake(0, 0, 0)
             sourceSize:MTLSizeMake(w, h, 1)
               toBuffer:dst->buffer
      destinationOffset:dstOff
 destinationBytesPerRow:bpr
destinationBytesPerImage:bpr * h];
}

// Inline blit shader: copies src (any readable format) → dst (read_write).
// Metal's texture interface always presents channels as float4(r,g,b,a)
// regardless of the underlying byte layout, so RGBA→BGRA copies are correct.
static constexpr const char* kBlitMSL =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "[[kernel]] void ngapi_blit(\n"
    "    texture2d<float, access::read>       src [[texture(0)]],\n"
    "    texture2d<float, access::read_write> dst [[texture(1)]],\n"
    "    uint2 gid [[thread_position_in_grid]])\n"
    "{\n"
    "    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;\n"
    "    dst.write(src.read(gid), gid);\n"
    "}\n";

static id<MTLComputePipelineState> getBlitPso(MetalDevice* md)
{
    if (md->blitPso) return md->blitPso;
    NSError* err = nil;
    NSString* msl = [NSString stringWithUTF8String:kBlitMSL];
    MTLCompileOptions* opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion3_1;
    id<MTLLibrary> lib = [md->device newLibraryWithSource:msl options:opts error:&err];
    if (!lib) { NSLog(@"getBlitPso: %@", err); return nil; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"ngapi_blit"];
    md->blitPso = [md->device newComputePipelineStateWithFunction:fn error:&err];
    if (!md->blitPso) NSLog(@"getBlitPso pipeline: %@", err);
    return md->blitPso;
}

void gpuBlitTexture(GpuCommandBuffer cb, GpuTexture dst, GpuTexture src)
{
    auto* md  = cb->device->metalDevice;
    auto  pso = getBlitPso(md);
    if (!pso || !src || !dst) return;

    auto enc = cb->compute();
    [enc setComputePipelineState:pso];
    [enc setTexture:src->texture atIndex:0];
    [enc setTexture:dst->texture atIndex:1];
    [enc useResource:src->texture usage:MTLResourceUsageRead];
    [enc useResource:dst->texture usage:MTLResourceUsageRead | MTLResourceUsageWrite];

    NSUInteger w = dst->texture.width;
    NSUInteger h = dst->texture.height;
    [enc dispatchThreadgroups:MTLSizeMake((w + 7) / 8, (h + 7) / 8, 1)
        threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
}

void gpuSetDepthStencilState(GpuCommandBuffer, GpuDepthStencilState) {}
void gpuSetBlendState(GpuCommandBuffer, GpuBlendState)               {}

void gpuSignalAfter(GpuCommandBuffer, STAGE, void*, uint64_t, SIGNAL)   {}
void gpuWaitBefore(GpuCommandBuffer, STAGE, void*, uint64_t, OP, HAZARD_FLAGS, uint64_t) {}

// ---------------------------------------------------------------------------
// Render pass / draw stubs
// ---------------------------------------------------------------------------

void gpuBeginRenderPass(GpuCommandBuffer, GpuRenderPassDesc)
{
    assert(false && "Metal: render pass not implemented in spike");
}

void gpuEndRenderPass(GpuCommandBuffer)
{
    assert(false && "Metal: render pass not implemented in spike");
}

void gpuDrawIndexedInstanced(GpuCommandBuffer, void*, void*, void*, uint32_t, uint32_t)
{
    assert(false && "Metal: draw not implemented in spike");
}

void gpuDrawIndexedInstancedIndirect(GpuCommandBuffer, void*, void*, void*, void*)
{
    assert(false && "Metal: draw not implemented in spike");
}

void gpuDrawIndexedInstancedIndirectMulti(GpuCommandBuffer, void*, uint32_t, void*, uint32_t, void*, void*, void*)
{
    assert(false && "Metal: draw not implemented in spike");
}

void gpuDrawMeshlets(GpuCommandBuffer, void*, void*, uint3)
{
    assert(false && "Metal: draw not implemented in spike");
}

void gpuDrawMeshletsIndirect(GpuCommandBuffer, void*, void*, void*)
{
    assert(false && "Metal: draw not implemented in spike");
}

// ---------------------------------------------------------------------------
// Surface / swapchain
// ---------------------------------------------------------------------------

#ifdef GPU_SURFACE_EXTENSION

Span<FORMAT> gpuSurfaceFormats(GpuDevice, GpuSurface) { return {}; }

GpuSwapchain gpuCreateSwapchain(GpuDevice device, GpuSurface surface, uint32_t framesInFlight)
{
    CAMetalLayer* layer = surface->layer;
    // Wire the device and cap the drawable count for CPU-side frame pacing.
    layer.device             = device->metalDevice->device;
    layer.maximumDrawableCount = framesInFlight < 2 ? 2 :
                                 framesInFlight > 3 ? 3 : framesInFlight;

    auto* sw             = new GpuSwapchain_T{};
    sw->layer            = layer;
    sw->device           = device;
    sw->framesInFlight   = framesInFlight;
    sw->swapchainTex.device = device;
    return sw;
}

void gpuDestroySwapchain(GpuSwapchain swapchain) { delete swapchain; }

GpuTextureDesc gpuSwapchainDesc(GpuSwapchain swapchain)
{
    GpuTextureDesc d{};
    d.type       = TEXTURE_2D;
    d.format     = FORMAT_BGRA8_SRGB; // closest NGAPI format for the BGRA8Unorm layer
    d.dimensions = { (uint32_t)swapchain->layer.drawableSize.width,
                     (uint32_t)swapchain->layer.drawableSize.height, 1 };
    return d;
}

// Acquire the next CAMetalDrawable and return its texture wrapped in a
// stable GpuTexture_T stored inside GpuSwapchain_T. Valid until the next
// call or gpuDestroySwapchain.
GpuTexture gpuSwapchainImage(GpuSwapchain swapchain)
{
    id<CAMetalDrawable> drawable = [swapchain->layer nextDrawable];
    swapchain->currentDrawable   = drawable;
    swapchain->swapchainTex.texture = drawable.texture;
    return &swapchain->swapchainTex;
}

// Present the drawable acquired by gpuSwapchainImage. Must be called after
// gpuSubmit — Metal schedules the present when the associated GPU work is done.
void gpuPresent(GpuSwapchain swapchain, GpuSemaphore /*semaphore*/, uint64_t /*value*/)
{
    if (swapchain->currentDrawable)
        [swapchain->currentDrawable present];
    swapchain->currentDrawable = nil;
}

#endif // GPU_SURFACE_EXTENSION

// ---------------------------------------------------------------------------
// Ray tracing stubs
// ---------------------------------------------------------------------------

#ifdef GPU_RAY_TRACING_EXTENSION

GpuAccelerationStructureSizes gpuAccelerationStructureSizes(GpuDevice, GpuAccelerationStructureDesc)
{
    assert(false && "Metal: acceleration structures not implemented in spike");
    return {};
}

GpuAccelerationStructure gpuCreateAccelerationStructure(GpuDevice, GpuAccelerationStructureDesc, void*, uint64_t)
{
    assert(false && "Metal: acceleration structures not implemented in spike");
    return nullptr;
}

void gpuBuildAccelerationStructures(GpuCommandBuffer, Span<GpuAccelerationStructure>, void*, MODE) {}

void gpuDestroyAccelerationStructure(GpuAccelerationStructure as) { delete as; }

#endif // GPU_RAY_TRACING_EXTENSION
