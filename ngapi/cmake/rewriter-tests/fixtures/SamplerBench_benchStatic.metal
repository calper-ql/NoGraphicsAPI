#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 1 "token paste"
constant ulong fc_staticLinearWrap_ngapiSlot_0 [[function_constant(0)]];
constant ulong staticLinearWrap_ngapiSlot_0 = is_function_constant_defined(fc_staticLinearWrap_ngapiSlot_0) ? fc_staticLinearWrap_ngapiSlot_0 : 5640617247111122947ULL;

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


#line 15 "samples/samplerbench/SamplerBench.slang"
HwSampler_0 staticLinearWrap_0()
{

#line 137 "ngapi/include/Sampler.h"
    return HwSampler_x24init_0(uint(staticLinearWrap_ngapiSlot_0));
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


#line 39 "samples/samplerbench/SamplerBench.slang"
struct EntryPointParams_0
{
    SamplerBenchData_0 device* data_0;
};


#line 39
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


#line 39 "samples/samplerbench/SamplerBench.slang"
[[kernel]] void benchStatic(uint3 threadId_0 [[thread_position_in_grid]], texture2d<float, access::sample>  textureHeap_1[], sampler  ngapiSamplerHeap_1[], texture2d<float, access::read_write>  rwTextureHeap_1[], EntryPointParams_0 constant* entryPointParams_1 [[buffer(0)]])
{

#line 39
    thread KernelContext_0 kernelContext_1;

#line 39
    (&kernelContext_1)->textureHeap_0 = textureHeap_1;

#line 39
    (&kernelContext_1)->ngapiSamplerHeap_0 = ngapiSamplerHeap_1;

#line 39
    (&kernelContext_1)->rwTextureHeap_0 = rwTextureHeap_1;

#line 39
    (&kernelContext_1)->entryPointParams_0 = entryPointParams_1;

    int _S2 = int(entryPointParams_1->data_0->imageSize_0.x);

#line 41
    int _S3 = int(entryPointParams_1->data_0->imageSize_0.y);

#line 41
    int2 imgSize_0 = int2(_S2, _S3);
    int _S4 = int(threadId_0.x);

#line 42
    int _S5 = int(threadId_0.y);

#line 42
    int2 pixelCoord_1 = int2(_S4, _S5);

#line 42
    bool _S6;
    if(_S4 >= _S2)
    {

#line 43
        _S6 = true;

#line 43
    }
    else
    {

#line 43
        _S6 = _S5 >= _S3;

#line 43
    }

#line 43
    if(_S6)
    {

#line 44
        return;
    }
    uint _S7 = entryPointParams_1->data_0->srcTexture_0;

#line 46
    texture2d<float, access::sample>  _S8[] = (&kernelContext_1)->textureHeap_0;
    HwSampler_0 _S9 = staticLinearWrap_0();
    float2 _S10 = float2(1.0)  / float2(imgSize_0);

    int r_0 = entryPointParams_1->data_0->kernelRadius_0;
    float4 _S11 = float4(0.0, 0.0, 0.0, 0.0);
    int _S12 = - entryPointParams_1->data_0->kernelRadius_0;

#line 52
    int ky_1 = _S12;

#line 52
    float4 acc_0 = _S11;

#line 52
    for(;;)
    {

#line 52
        if(ky_1 <= r_0)
        {
        }
        else
        {

#line 52
            break;
        }

#line 52
        int kx_1 = _S12;
        for(;;)
        {

#line 53
            if(kx_1 <= r_0)
            {
            }
            else
            {

#line 53
                break;
            }

#line 54
            float2 _S13 = tapUV_0(pixelCoord_1, kx_1, ky_1, _S10);

#line 54
            thread HwSampler_0 _S14 = _S9;

#line 54
            float4 _S15 = HwSampler_SampleLevel_0(&_S14, _S8[_S7], _S13, 0.0, &kernelContext_1);

#line 54
            float4 acc_1 = acc_0 + _S15;

#line 53
            kx_1 = kx_1 + int(1);

#line 53
            acc_0 = acc_1;

#line 53
        }

#line 52
        ky_1 = ky_1 + int(1);

#line 52
    }



    int _S16 = int(2) * r_0 + int(1);
    (&kernelContext_1)->rwTextureHeap_0[entryPointParams_1->data_0->dstTexture_0].write(acc_0 / float4(float(_S16 * _S16)) ,uint2(pixelCoord_1));
    return;
}

