#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 1 "token paste"
constant ulong fc_linearWrap_ngapiSlot_0 [[function_constant(0)]];
constant ulong linearWrap_ngapiSlot_0 = is_function_constant_defined(fc_linearWrap_ngapiSlot_0) ? fc_linearWrap_ngapiSlot_0 : 5640617247111122947ULL;

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


#line 3 "samples/common/TAA.slang"
HwSampler_0 linearWrap_0()
{

#line 137 "ngapi/include/Sampler.h"
    return HwSampler_x24init_0(uint(linearWrap_ngapiSlot_0));
}


#line 6 "samples/common/TAA.h"
struct TAAData_0
{
    uint width_0;
    uint height_0;
    uint frame_0;
    uint srcColor_0;
    uint srcHistory_0;
    uint srcDepth_0;
    uint srcMotionVectors_0;
    uint dstTexture_0;
    float2 jitter_0;
};


#line 5 "samples/common/TAA.slang"
struct EntryPointParams_0
{
    TAAData_0 device* data_0;
};


#line 5
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


#line 5 "samples/common/TAA.slang"
[[kernel]] void main_0(uint3 threadId_0 [[thread_position_in_grid]], texture2d<float, access::sample>  textureHeap_1[], sampler  ngapiSamplerHeap_1[], texture2d<float, access::read_write>  rwTextureHeap_1[], EntryPointParams_0 constant* entryPointParams_1 [[buffer(0)]])
{

#line 5
    thread KernelContext_0 kernelContext_1;

#line 5
    (&kernelContext_1)->textureHeap_0 = textureHeap_1;

#line 5
    (&kernelContext_1)->ngapiSamplerHeap_0 = ngapiSamplerHeap_1;

#line 5
    (&kernelContext_1)->rwTextureHeap_0 = rwTextureHeap_1;

#line 5
    (&kernelContext_1)->entryPointParams_0 = entryPointParams_1;

    uint width_1 = entryPointParams_1->data_0->width_0;
    uint height_1 = entryPointParams_1->data_0->height_0;

    uint _S2 = threadId_0.x;

#line 10
    bool validUV_0;

#line 10
    if(_S2 >= (entryPointParams_1->data_0->width_0))
    {

#line 10
        validUV_0 = true;

#line 10
    }
    else
    {

#line 10
        validUV_0 = (threadId_0.y) >= height_1;

#line 10
    }

#line 10
    if(validUV_0)
    {

#line 11
        return;
    }
    float2 _S3 = float2(float(width_1), float(height_1));

#line 13
    float2 uv_1 = float2(float(_S2) + 0.5, float(threadId_0.y) + 0.5) / _S3;
    float2 _S4 = float2(1.0)  / _S3;

    uint _S5 = entryPointParams_1->data_0->srcColor_0;

#line 16
    texture2d<float, access::sample>  _S6[] = (&kernelContext_1)->textureHeap_0;
    HwSampler_0 sampler_0 = linearWrap_0();

#line 17
    thread HwSampler_0 _S7 = sampler_0;

#line 17
    float4 _S8 = HwSampler_SampleLevel_0(&_S7, (&kernelContext_1)->textureHeap_0[_S5], uv_1, 0.0, &kernelContext_1);


    uint _S9 = entryPointParams_1->data_0->srcMotionVectors_0;

#line 20
    thread HwSampler_0 _S10 = sampler_0;

#line 20
    float4 _S11 = HwSampler_SampleLevel_0(&_S10, (&kernelContext_1)->textureHeap_0[_S9], uv_1, 0.0, &kernelContext_1);

#line 20
    float2 motionVectors_0 = _S11.xy;

#line 20
    float4 neighborMin_0 = _S8;

#line 20
    float4 neighborMax_0 = _S8;

#line 20
    int y_0 = int(-1);

#line 26
    for(;;)
    {

#line 26
        if(y_0 <= int(1))
        {
        }
        else
        {

#line 26
            break;
        }

#line 26
        int x_0 = int(-1);

        for(;;)
        {

#line 28
            if(x_0 <= int(1))
            {
            }
            else
            {

#line 28
                break;
            }
            float2 _S12 = uv_1 + float2(float(x_0), float(y_0)) * _S4;

#line 30
            thread HwSampler_0 _S13 = sampler_0;

#line 30
            float4 _S14 = HwSampler_SampleLevel_0(&_S13, _S6[_S5], _S12, 0.0, &kernelContext_1);
            float4 _S15 = min(neighborMin_0, _S14);
            float4 _S16 = max(neighborMax_0, _S14);

#line 28
            int _S17 = x_0 + int(1);

#line 28
            neighborMin_0 = _S15;

#line 28
            neighborMax_0 = _S16;

#line 28
            x_0 = _S17;

#line 28
        }

#line 26
        y_0 = y_0 + int(1);

#line 26
    }

#line 37
    float2 reprojectedUV_0 = uv_1 - motionVectors_0 * float2(0.5) ;
    uint _S18 = entryPointParams_1->data_0->srcHistory_0;

#line 38
    thread HwSampler_0 _S19 = sampler_0;

#line 38
    float4 _S20 = HwSampler_SampleLevel_0(&_S19, (&kernelContext_1)->textureHeap_0[_S18], reprojectedUV_0, 0.0, &kernelContext_1);


    float4 historyColor_0 = clamp(_S20, neighborMin_0, neighborMax_0);

    float _S21 = reprojectedUV_0.x;

#line 43
    if(_S21 >= 0.0)
    {

#line 43
        validUV_0 = _S21 <= 1.0;

#line 43
    }
    else
    {

#line 43
        validUV_0 = false;

#line 43
    }

#line 43
    if(validUV_0)
    {

#line 43
        validUV_0 = (reprojectedUV_0.y) >= 0.0;

#line 43
    }
    else
    {

#line 43
        validUV_0 = false;

#line 43
    }
    if(validUV_0)
    {

#line 44
        validUV_0 = (reprojectedUV_0.y) <= 1.0;

#line 44
    }
    else
    {

#line 44
        validUV_0 = false;

#line 44
    }



    if((entryPointParams_1->data_0->frame_0) == 0U)
    {

#line 48
        validUV_0 = true;

#line 48
    }
    else
    {

#line 48
        validUV_0 = !validUV_0;

#line 48
    }

#line 48
    float blendFactor_0;

#line 48
    if(validUV_0)
    {

#line 48
        blendFactor_0 = 1.0;

#line 48
    }
    else
    {

#line 48
        blendFactor_0 = 0.05000000074505806;

#line 48
    }

#line 56
    (&kernelContext_1)->rwTextureHeap_0[entryPointParams_1->data_0->dstTexture_0].write(mix(historyColor_0, _S8, float4(blendFactor_0) ),threadId_0.xy);
    return;
}

