#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 1 "token paste"
constant ulong fc_staticNearestClamp_ngapiSlot_0 [[function_constant(1)]];
constant ulong staticNearestClamp_ngapiSlot_0 = is_function_constant_defined(fc_staticNearestClamp_ngapiSlot_0) ? fc_staticNearestClamp_ngapiSlot_0 : 5640617247111123528ULL;

#line 100 "ngapi/include/Sampler.h"
struct HwSampler_0
{
    uint slot_0;
};


#line 100
HwSampler_0 HwSampler_x24init_0(uint slot_1)
{

#line 100
    thread HwSampler_0 _S1;

    (&_S1)->slot_0 = slot_1;

#line 100
    return _S1;
}


#line 16 "samples/samplerbench/SamplerBench.slang"
HwSampler_0 staticNearestClamp_0()
{

#line 137 "ngapi/include/Sampler.h"
    return HwSampler_x24init_0(uint(staticNearestClamp_ngapiSlot_0));
}


#line 10 "samples/samplerbench/SamplerBench.slang"
float2 tapUV_0(int2 pixelCoord_0, int kx_0, int ky_0, float2 invSize_0)
{
    return (float2(pixelCoord_0) + float2(0.5)  + float2(float(kx_0), float(ky_0)) * float2(0.70999997854232788) ) * invSize_0;
}


#line 6 "samples/samplerbench/SamplerBench.h"
struct SamplerBenchData_0
{
    uint2 imageSize_0;
    uint srcTexture_0;
    uint dstTexture_0;
    int kernelRadius_0;
};


#line 81 "samples/samplerbench/SamplerBench.slang"
struct EntryPointParams_0
{
    SamplerBenchData_0 device* data_0;
};


#line 81
struct KernelContext_0
{
    texture2d<float, access::sample>  textureHeap_0[];
    sampler  ngapiSamplerHeap_0[];
    texture2d<float, access::read_write>  rwTextureHeap_0[];
    EntryPointParams_0 constant* entryPointParams_0;
};


#line 109 "ngapi/include/Sampler.h"
float4 HwSampler_SampleLevel_0(const HwSampler_0 thread* this_0, texture2d<float, access::sample> tex_0, float2 uv_0, float lod_0, KernelContext_0 thread* kernelContext_0)
{
    return ((tex_0).sample((kernelContext_0->ngapiSamplerHeap_0[this_0->slot_0]), (uv_0), level((lod_0))));
}


#line 81 "samples/samplerbench/SamplerBench.slang"
[[kernel]] void benchStaticNearest(uint3 threadId_0 [[thread_position_in_grid]], texture2d<float, access::sample>  textureHeap_1[], sampler  ngapiSamplerHeap_1[], texture2d<float, access::read_write>  rwTextureHeap_1[], EntryPointParams_0 constant* entryPointParams_1 [[buffer(0)]])
{

#line 81
    thread KernelContext_0 kernelContext_1;

#line 81
    (&kernelContext_1)->textureHeap_0 = textureHeap_1;

#line 81
    (&kernelContext_1)->ngapiSamplerHeap_0 = ngapiSamplerHeap_1;

#line 81
    (&kernelContext_1)->rwTextureHeap_0 = rwTextureHeap_1;

#line 81
    (&kernelContext_1)->entryPointParams_0 = entryPointParams_1;

    int _S2 = int(entryPointParams_1->data_0->imageSize_0.x);

#line 83
    int _S3 = int(entryPointParams_1->data_0->imageSize_0.y);

#line 83
    int2 imgSize_0 = int2(_S2, _S3);
    int _S4 = int(threadId_0.x);

#line 84
    int _S5 = int(threadId_0.y);

#line 84
    int2 pixelCoord_1 = int2(_S4, _S5);

#line 84
    bool _S6;
    if(_S4 >= _S2)
    {

#line 85
        _S6 = true;

#line 85
    }
    else
    {

#line 85
        _S6 = _S5 >= _S3;

#line 85
    }

#line 85
    if(_S6)
    {

#line 86
        return;
    }
    uint _S7 = entryPointParams_1->data_0->srcTexture_0;

#line 88
    texture2d<float, access::sample>  _S8[] = (&kernelContext_1)->textureHeap_0;
    HwSampler_0 _S9 = staticNearestClamp_0();
    float2 _S10 = float2(1.0)  / float2(imgSize_0);

    int r_0 = entryPointParams_1->data_0->kernelRadius_0;
    float4 _S11 = float4(0.0, 0.0, 0.0, 0.0);
    int _S12 = - entryPointParams_1->data_0->kernelRadius_0;

#line 94
    int ky_1 = _S12;

#line 94
    float4 acc_0 = _S11;

#line 94
    for(;;)
    {

#line 94
        if(ky_1 <= r_0)
        {
        }
        else
        {

#line 94
            break;
        }

#line 94
        int kx_1 = _S12;
        for(;;)
        {

#line 95
            if(kx_1 <= r_0)
            {
            }
            else
            {

#line 95
                break;
            }

#line 96
            float2 _S13 = tapUV_0(pixelCoord_1, kx_1, ky_1, _S10);

#line 96
            thread HwSampler_0 _S14 = _S9;

#line 96
            float4 _S15 = HwSampler_SampleLevel_0(&_S14, _S8[_S7], _S13, 0.0, &kernelContext_1);

#line 96
            float4 acc_1 = acc_0 + _S15;

#line 95
            kx_1 = kx_1 + int(1);

#line 95
            acc_0 = acc_1;

#line 95
        }

#line 94
        ky_1 = ky_1 + int(1);

#line 94
    }



    int _S16 = int(2) * r_0 + int(1);
    (&kernelContext_1)->rwTextureHeap_0[entryPointParams_1->data_0->dstTexture_0].write(acc_0 / float4(float(_S16 * _S16)) ,uint2(pixelCoord_1));
    return;
}

