#include "stb_image.h"
#include "stb_image_write.h"

#include <cstdio>
#include <cstring>

#include "window.h"

#include "Compute.h"
#include "Utilities.h"

int main()
{
    gpuCreateInstance();
    auto device = gpuCreateDevice(0);

    LinearAllocator allocator(device);

    auto queue     = gpuCreateQueue(device);
    auto semaphore = gpuCreateSemaphore(device, 0);

#ifdef GPU_METAL_BACKEND
    auto computeIR = loadIR("shaders/compute/Compute.metal");
#else
    auto computeIR = loadIR("shaders/compute/Compute.spv");
#endif
    auto pipeline = gpuCreateComputePipeline(device, ByteSpan(computeIR.data(), computeIR.size()));

    auto textureHeap = allocator.allocate<GpuTextureDescriptor>(1024);

    // Load input image
    int width, height, channels;
    stbi_uc* inputImage = stbi_load("assets/Default.png", &width, &height, &channels, 4);
    if (!inputImage)
    {
        fprintf(stderr, "compute: failed to load assets/Default.png\n");
        return 1;
    }

    auto upload = allocator.allocate<uint8_t>((size_t)width * height * 4);
    memcpy(upload.cpu, inputImage, (size_t)width * height * 4);

    GpuTextureDesc textureDesc{
        .type       = TEXTURE_2D,
        .dimensions = { (uint32_t)width, (uint32_t)height, 1 },
        .format     = FORMAT_RGBA8_UNORM,
        .usage      = (USAGE_FLAGS)(USAGE_SAMPLED | USAGE_TRANSFER_DST)
    };

    GpuTextureSizeAlign textureSizeAlign = gpuTextureSizeAlign(device, textureDesc);
    void* texturePtr = gpuMalloc(device, textureSizeAlign.size, MEMORY_GPU);
    auto texture = gpuCreateTexture(device, textureDesc, texturePtr);

    GpuTextureDesc outputTextureDesc{
        .type       = TEXTURE_2D,
        .dimensions = { (uint32_t)width, (uint32_t)height, 1 },
        .format     = FORMAT_RGBA8_UNORM,
        .usage      = (USAGE_FLAGS)(USAGE_STORAGE | USAGE_TRANSFER_SRC)
    };

    void* outputPtr = gpuMalloc(device, textureSizeAlign.size, MEMORY_GPU);
    auto outputTexture = gpuCreateTexture(device, outputTextureDesc, outputPtr);

    textureHeap.cpu[0] = gpuTextureViewDescriptor  (texture,       GpuViewDesc{ .format = FORMAT_RGBA8_UNORM });
    textureHeap.cpu[1] = gpuRWTextureViewDescriptor(outputTexture, GpuViewDesc{ .format = FORMAT_RGBA8_UNORM });

    // Upload input image to GPU texture
    {
        auto cb = gpuStartCommandRecording(queue);
        gpuCopyToTexture(cb, upload.gpu, texture);
        gpuSubmit(queue, Span<GpuCommandBuffer>(&cb, 1), semaphore, 1);
        gpuWaitSemaphore(semaphore, 1);
    }

    // Per-band tint data
    const uint32_t tintCount = 4;
    auto tints = allocator.allocate<Tint>(tintCount);
    tints.cpu[0] = { { 1.0f, 0.5f, 0.5f }, 0.6f }; // red
    tints.cpu[1] = { { 0.5f, 1.0f, 0.5f }, 0.6f }; // green
    tints.cpu[2] = { { 0.5f, 0.5f, 1.0f }, 0.6f }; // blue
    tints.cpu[3] = { { 1.0f, 1.0f, 0.5f }, 0.6f }; // yellow

    auto data = allocator.allocate<ComputeData>(1);
    data.cpu->imageSize  = { (uint32_t)width, (uint32_t)height };
    data.cpu->srcTexture = 0;
    data.cpu->dstTexture = 1;
    data.cpu->tints      = tints.gpu;
    data.cpu->tintCount  = tintCount;

    // Dispatch compute
    {
        auto cb = gpuStartCommandRecording(queue);
        gpuSetPipeline(cb, pipeline);
        gpuSetActiveTextureHeapPtr(cb, textureHeap.gpu);
        gpuDispatch(cb, data.gpu,
            { (uint32_t)(width / 16), (uint32_t)(height / 16), 1 });
        gpuSubmit(queue, Span<GpuCommandBuffer>(&cb, 1), semaphore, 2);
        gpuWaitSemaphore(semaphore, 2);
    }

    // Windowed path: blit the compute result to the swapchain each frame.
    const uint FRAMES_IN_FLIGHT = 2;
    auto window    = ngapi::createWindow("Compute", width, height);
    auto surface   = ngapi::createSurface(window);
    auto swapchain = gpuCreateSwapchain(device, surface, FRAMES_IN_FLIGHT);

    gpuDestroySemaphore(semaphore);
    semaphore = gpuCreateSemaphore(device, 0);
    uint64_t nextFrame = 1;

    while (!ngapi::shouldClose(window))
    {
        ngapi::pollEvents(window);

        if (nextFrame > FRAMES_IN_FLIGHT)
            gpuWaitSemaphore(semaphore, nextFrame - FRAMES_IN_FLIGHT);

        auto cb    = gpuStartCommandRecording(queue);
        auto image = gpuSwapchainImage(swapchain);
        gpuBlitTexture(cb, image, outputTexture);
        gpuSubmit(queue, Span<GpuCommandBuffer>(&cb, 1), semaphore, nextFrame);
        gpuPresent(swapchain, semaphore, nextFrame++);
    }

    gpuWaitSemaphore(semaphore, nextFrame - 1);
    gpuDestroySwapchain(swapchain);
    ngapi::destroySurface(window, surface);
    ngapi::destroyWindow(window);

    stbi_image_free(inputImage);
    allocator.reset();

    gpuDestroySemaphore(semaphore);
    gpuDestroyTexture(texture);
    gpuDestroyTexture(outputTexture);
    gpuFreePipeline(pipeline);
    gpuFree(device, texturePtr);
    gpuFree(device, outputPtr);
    gpuDestroyQueue(queue);
    gpuDestroyDevice(device);
    gpuDestroyInstance();

    return 0;
}
