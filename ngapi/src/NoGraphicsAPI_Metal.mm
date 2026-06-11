// Metal backend — covers gpuCreateDevice, gpuMalloc, gpuDispatch and enough
// glue to run the headless samples (multiplegpus, learning). Windowed /
// surface / raytracing paths are unimplemented stubs.
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
//      Xcode pre-compilation (xcrun metal → .air → xcrun metallib) is a
//      future improvement.
//
//   4. Timeline semaphores: mapped 1:1 onto MTLSharedEvent, which supports
//      signaling at a value and notifying a listener at a threshold — identical
//      semantics to VkSemaphoreTypeTimeline.
//
//   5. MEMORY_GPU: Private storage mode has no CPU-visible pointer. We return
//      the MTLBuffer's GPU address cast to void* (same convention as the Vulkan
//      backend's MEMORY_GPU path, which returns the VkDeviceAddress).

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>

#define GPU_EXPOSE_INTERNAL
#include "NoGraphicsAPI.h"

#include <algorithm>
#include <mutex>
#include <vector>
#include <cassert>
#include <cstring>

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

    std::mutex             allocMutex;
    std::vector<MetalAllocation> allocations;

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

struct GpuDevice_T            { MetalDevice*                metalDevice = nullptr; };
struct GpuPipeline_T
{
    id<MTLComputePipelineState> pso;
    GpuDevice  device;
    NSUInteger entryParamsIndex = 0;   // [[buffer(N)]] index for EntryPointParams
    NSUInteger threadgroupSize  = 64;  // threads per group; matches [numthreads] in the shader
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
struct GpuSurface_T   { void* placeholder; };
struct GpuSwapchain_T { void* placeholder; };
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

GpuSurface gpuCreateSurface(void* /*vulkanSurface*/)
{
    return new GpuSurface_T{};
}

void* gpuVulkanSurface(GpuSurface /*surface*/) { return nullptr; }

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

    // Match the Vulkan convention: MEMORY_GPU returns the GPU address cast to
    // void* (not a CPU pointer), everything else returns the CPU pointer.
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
// Textures (stubs — not implemented for this spike)
// ---------------------------------------------------------------------------

GpuTextureSizeAlign gpuTextureSizeAlign(GpuDevice, GpuTextureDesc)
{
    assert(false && "Metal: textures not implemented in spike");
    return {};
}

GpuTexture gpuCreateTexture(GpuDevice, GpuTextureDesc, void*)
{
    assert(false && "Metal: textures not implemented in spike");
    return nullptr;
}

void gpuDestroyTexture(GpuTexture) {}

GpuTextureDescriptor gpuTextureViewDescriptor(GpuTexture, GpuViewDesc)
{
    return {};
}

GpuTextureDescriptor gpuRWTextureViewDescriptor(GpuTexture, GpuViewDesc)
{
    return {};
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

    if (isMetallib)
    {
        dispatch_data_t data = dispatch_data_create(
            ir.data(), ir.size(), nil, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        lib = [md->device newLibraryWithData:data error:&err];
    }
    else
    {
        // Treat bytes as UTF-8 MSL source.
        // MSL 3.1 (macOS 14+) is required for Slang's calling convention:
        // loading a device* from constant memory and dereferencing it directly
        // (the EntryPointParams pattern) is only valid in MSL 3.0+.
        NSString* src = [[NSString alloc] initWithBytes:ir.data()
                                                 length:ir.size()
                                               encoding:NSUTF8StringEncoding];
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.languageVersion = MTLLanguageVersion3_1;
        lib = [md->device newLibraryWithSource:src options:opts error:&err];
    }

    if (!lib)
    {
        NSLog(@"gpuCreateComputePipeline: library error: %@", err);
        return nullptr;
    }

    NSString* entryName = entry ? [NSString stringWithUTF8String:entry] : @"main0";
    id<MTLFunction> fn = [lib newFunctionWithName:entryName];
    if (!fn)
    {
        NSLog(@"gpuCreateComputePipeline: entry '%s' not found", entry ? entry : "main0");
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

    // Find the [[buffer(N)]] index that Slang assigned to EntryPointParams.
    // Single-entry .metal files always produce N=0; multi-entry files assign
    // N by position. Reflection is the authoritative source.
    NSUInteger paramsIndex = 0;
    for (id<MTLBinding> b in refl.bindings)
    {
        if (b.type == MTLBindingTypeBuffer && [b.name hasPrefix:@"entryPointParams"])
        {
            paramsIndex = b.index;
            break;
        }
    }

    return new GpuPipeline_T{ pso, device, paramsIndex, 64 };
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
    // Encode pipeline state immediately; the compute encoder is kept alive
    // until another encoder type is needed or the command buffer is submitted.
    auto enc = cb->compute();
    [enc setComputePipelineState:pipeline->pso];
}

void gpuSetActiveTextureHeapPtr(GpuCommandBuffer /*cb*/, void* /*ptrGpu*/)
{
    // No-op for the spike — texture heap binding is not implemented.
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------
//
// Slang's MSL output wraps entry-point parameters in
//   `EntryPointParams_N constant* entryPointParams_N [[buffer(N)]]`
// where the struct holds a device pointer to the user's data struct.
// We pass the 8-byte GPU address via setBytes — the GPU reads it as a device
// pointer and dereferences it directly (Apple Silicon unified address space).
// N is stored in pipeline->entryParamsIndex (discovered via reflection at
// pipeline creation time).

void gpuDispatch(GpuCommandBuffer cb, void* dataGpu, uint3 gridDimensions)
{
    auto* md  = cb->device->metalDevice;
    auto  enc = cb->compute();

    // Pack the 8-byte GPU address into a device-accessible buffer.
    // The MSL kernel declares `device EntryPointParams_N* p [[buffer(N)]]`
    // where EntryPointParams contains a device* to the actual data struct.
    uint64_t addr = reinterpret_cast<uint64_t>(dataGpu);
    id<MTLBuffer> paramBuf = [md->device newBufferWithBytes:&addr
                                                     length:sizeof(addr)
                                                    options:MTLResourceStorageModeShared];
    [enc setBuffer:paramBuf
            offset:0
           atIndex:cb->currentPipeline->entryParamsIndex];

    // Metal requires every buffer accessed via device-pointer indirection to be
    // declared to the encoder so the GPU can map it. Declare all live allocations
    // read+write. (An MTLHeap would be more efficient; this is correct for Phase 1.)
    {
        std::lock_guard lock(md->allocMutex);
        for (auto& a : md->allocations)
            [enc useResource:a.buffer usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    }
    [enc useResource:paramBuf usage:MTLResourceUsageRead];

    MTLSize grid        = MTLSizeMake(gridDimensions.x, gridDimensions.y, gridDimensions.z);
    MTLSize threadgroup = MTLSizeMake(cb->currentPipeline->threadgroupSize, 1, 1);

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
                                 threadsPerThreadgroup:MTLSizeMake(cb->currentPipeline->threadgroupSize, 1, 1)];
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
// Commands — blit / barrier
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
    // Between two compute encoders a memoryBarrierWithScope would be needed —
    // not required for the spike's linear workloads.
    (void)cb;
}

void gpuCopyToTexture(GpuCommandBuffer, void*, GpuTexture)
{
    assert(false && "Metal: gpuCopyToTexture not implemented in spike");
}

void gpuCopyFromTexture(GpuCommandBuffer, void*, GpuTexture)
{
    assert(false && "Metal: gpuCopyFromTexture not implemented in spike");
}

void gpuBlitTexture(GpuCommandBuffer, GpuTexture, GpuTexture)
{
    assert(false && "Metal: gpuBlitTexture not implemented in spike");
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
// Surface / swapchain stubs
// ---------------------------------------------------------------------------

#ifdef GPU_SURFACE_EXTENSION

Span<FORMAT> gpuSurfaceFormats(GpuDevice, GpuSurface) { return {}; }

GpuSwapchain gpuCreateSwapchain(GpuDevice, GpuSurface, uint32_t)
{
    assert(false && "Metal: swapchain not implemented in spike");
    return nullptr;
}

void gpuDestroySwapchain(GpuSwapchain swapchain) { delete swapchain; }

GpuTextureDesc gpuSwapchainDesc(GpuSwapchain)       { return {}; }
GpuTexture     gpuSwapchainImage(GpuSwapchain)      { return nullptr; }
void           gpuPresent(GpuSwapchain, GpuSemaphore, uint64_t) {}

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
