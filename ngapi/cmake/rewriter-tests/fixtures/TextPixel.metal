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


#line 3 "samples/common/TextPixel.slang"
HwSampler_0 linearWrap_0()
{

#line 137 "ngapi/include/Sampler.h"
    return HwSampler_x24init_0(uint(linearWrap_ngapiSlot_0));
}


#line 6 "samples/common/Text.h"
struct TextVertexData_0
{
    uint width_0;
    uint height_0;
    uint textWidth_0;
    uint textHeight_0;
    uint atlasWidth_0;
    uint atlasHeight_0;
    uchar device* text_0;
};

struct TextPixelData_0
{
    uint atlas_0;
};


#line 11 "samples/common/TextPixel.slang"
struct EntryPointParams_0
{
    TextVertexData_0 device* _0;
    TextPixelData_0 device* data_0;
};


#line 11
struct KernelContext_0
{
    texture2d<float, access::sample>  textureHeap_0[];
    sampler  ngapiSamplerHeap_0[];
    EntryPointParams_0 constant* entryPointParams_0;
};


#line 109 "ngapi/include/Sampler.h"
float4 HwSampler_SampleLevel_0(const HwSampler_0 thread* this_0, texture2d<float, access::sample> tex_0, float2 uv_0, float lod_0, KernelContext_0 thread* kernelContext_0)
{
    return ((tex_0).sample((kernelContext_0->ngapiSamplerHeap_0[this_0->slot_0]), (uv_0), level((lod_0))));
}


#line 111
struct pixelOutput_0
{
    float4 output_0 [[color(0)]];
};


#line 111
struct pixelInput_0
{
    float2 uv_1 [[user(_SLANG_ATTR)]];
};


#line 11 "samples/common/TextPixel.slang"
[[fragment]] pixelOutput_0 main_0(pixelInput_0 _S2 [[stage_in]], float4 position_0 [[position]], texture2d<float, access::sample>  textureHeap_1[], sampler  ngapiSamplerHeap_1[], EntryPointParams_0 constant* entryPointParams_1 [[buffer(0)]])
{

#line 11
    thread KernelContext_0 kernelContext_1;

#line 11
    (&kernelContext_1)->textureHeap_0 = textureHeap_1;

#line 11
    (&kernelContext_1)->ngapiSamplerHeap_0 = ngapiSamplerHeap_1;

#line 11
    (&kernelContext_1)->entryPointParams_0 = entryPointParams_1;

    uint _S3 = entryPointParams_1->data_0->atlas_0;

#line 13
    thread HwSampler_0 _S4 = linearWrap_0();

#line 13
    float4 _S5 = HwSampler_SampleLevel_0(&_S4, textureHeap_1[_S3], _S2.uv_1, 0.0, &kernelContext_1);
    if((_S5.w) == 0.0)
    {
        discard_fragment();

#line 14
    }

#line 14
    pixelOutput_0 _S6 = { _S5 };



    return _S6;
}

