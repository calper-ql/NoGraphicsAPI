#ifndef SAMPLES_SHADER_SAMPLER_BENCH_H
#define SAMPLES_SHADER_SAMPLER_BENCH_H

#include "NoGraphicsAPI.h"

struct alignas(16) SamplerBenchData
{
    uint2 imageSize;
    uint srcTexture;
    uint dstTexture;
    int kernelRadius; // taps per pixel = (2*kernelRadius + 1)^2
};

#endif // SAMPLES_SHADER_SAMPLER_BENCH_H
