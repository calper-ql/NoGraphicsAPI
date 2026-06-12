#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

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


#line 8 "samples/compute/Compute.h"
struct Tint_0
{
    float4 color_0;
    float strength_0;
};

struct ComputeData_0
{
    uint2 imageSize_0;
    uint srcTexture_0;
    uint dstTexture_0;
    Tint_0 device* tints_0;
    uint tintCount_0;
};


#line 3 "samples/compute/Compute.slang"
struct EntryPointParams_0
{
    ComputeData_0 device* data_0;
};


#line 3
struct KernelContext_0
{
    texture2d<float, access::sample>  textureHeap_0[];
    texture2d<float, access::read_write>  rwTextureHeap_0[];
    sampler  ngapiSamplerHeap_0[];
    EntryPointParams_0 constant* entryPointParams_0;
};


#line 109 "ngapi/include/Sampler.h"
float4 HwSampler_SampleLevel_0(const HwSampler_0 thread* this_0, texture2d<float, access::sample> tex_0, float2 uv_0, float lod_0, KernelContext_0 thread* kernelContext_0)
{
    return ((tex_0).sample((kernelContext_0->ngapiSamplerHeap_0[this_0->slot_0]), (uv_0), level((lod_0))));
}


#line 3 "samples/compute/Compute.slang"
[[kernel]] void main_0(uint3 threadId_0 [[thread_position_in_grid]], texture2d<float, access::sample>  textureHeap_1[], texture2d<float, access::read_write>  rwTextureHeap_1[], sampler  ngapiSamplerHeap_1[], EntryPointParams_0 constant* entryPointParams_1 [[buffer(0)]])
{

#line 3
    thread KernelContext_0 kernelContext_1;

#line 3
    (&kernelContext_1)->textureHeap_0 = textureHeap_1;

#line 3
    (&kernelContext_1)->rwTextureHeap_0 = rwTextureHeap_1;

#line 3
    (&kernelContext_1)->ngapiSamplerHeap_0 = ngapiSamplerHeap_1;

#line 3
    (&kernelContext_1)->entryPointParams_0 = entryPointParams_1;

    int _S2 = int(entryPointParams_1->data_0->imageSize_0.x);

#line 5
    int _S3 = int(entryPointParams_1->data_0->imageSize_0.y);

#line 5
    int2 imgSize_0 = int2(_S2, _S3);
    int _S4 = int(threadId_0.x);

#line 6
    int _S5 = int(threadId_0.y);

#line 6
    int2 pixelCoord_0 = int2(_S4, _S5);

#line 6
    bool _S6;

    if(_S4 >= _S2)
    {

#line 8
        _S6 = true;

#line 8
    }
    else
    {

#line 8
        _S6 = _S5 >= _S3;

#line 8
    }

#line 8
    if(_S6)
    {

#line 9
        return;
    }
    uint _S7 = entryPointParams_1->data_0->srcTexture_0;

#line 11
    texture2d<float, access::sample>  _S8[] = (&kernelContext_1)->textureHeap_0;
    uint _S9 = entryPointParams_1->data_0->dstTexture_0;

#line 12
    texture2d<float, access::read_write>  _S10[] = (&kernelContext_1)->rwTextureHeap_0;

#line 143 "ngapi/include/Sampler.h"
    HwSampler_0 _S11 = HwSampler_x24init_0(1308627531U);

#line 18 "samples/compute/Compute.slang"
    thread float4 color_1 = float4(0.0, 0.0, 0.0, 0.0);

#line 18
    int ky_0 = int(-16);


    for(;;)
    {

#line 21
        if(ky_0 <= int(16))
        {
        }
        else
        {

#line 21
            break;
        }

#line 21
        int kx_0 = int(-16);

        for(;;)
        {

#line 23
            if(kx_0 <= int(16))
            {
            }
            else
            {

#line 23
                break;
            }

            int2 sampleCoord_0 = clamp(pixelCoord_0 + int2(kx_0, ky_0), int2(int(0), int(0)), imgSize_0 - int2(int(1), int(1)));
            float2 uv_1 = (float2(float(sampleCoord_0.x), float(sampleCoord_0.y)) + float2(0.5) ) / float2(float(_S2), float(_S3));

#line 27
            thread HwSampler_0 _S12 = _S11;

#line 27
            float4 _S13 = HwSampler_SampleLevel_0(&_S12, _S8[_S7], uv_1, 0.0, &kernelContext_1);
            color_1 = color_1 + _S13;

#line 23
            kx_0 = kx_0 + int(1);

#line 23
        }

#line 21
        ky_0 = ky_0 + int(1);

#line 21
    }

#line 32
    color_1 = color_1 / float4(pow(33.0, 2.0)) ;



    uint _S14 = uint(_S4) * entryPointParams_1->data_0->tintCount_0 / uint(_S2);
    Tint_0 device* _S15 = entryPointParams_1->data_0->tints_0 + min(_S14, entryPointParams_1->data_0->tintCount_0 - 1U);
    color_1.xyz = mix(color_1.xyz, color_1.xyz * (*_S15).color_0.xyz, float3((*_S15).strength_0) );

    _S10[_S9].write(color_1,uint2(pixelCoord_0));
    return;
}

