#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 1 "token paste"
constant ulong fc_aniso16Wrap_ngapiSlot_0 [[function_constant(0)]];
constant ulong aniso16Wrap_ngapiSlot_0 = is_function_constant_defined(fc_aniso16Wrap_ngapiSlot_0) ? fc_aniso16Wrap_ngapiSlot_0 : 5640617247111184391ULL;

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


#line 6 "samples/graphics/Pixel.slang"
HwSampler_0 aniso16Wrap_0()
{

#line 137 "ngapi/include/Sampler.h"
    return HwSampler_x24init_0(uint(aniso16Wrap_ngapiSlot_0));
}


#line 106
struct _MatrixStorage_float4x4_ColMajornatural_0
{
    array<float4, int(4)> data_0;
};


#line 14 "samples/graphics/Graphics.h"
struct Instance_natural_0
{
    _MatrixStorage_float4x4_ColMajornatural_0 model_0;
    _MatrixStorage_float4x4_ColMajornatural_0 prevModel_0;
};


#line 14
struct VertexData_natural_0
{
    _MatrixStorage_float4x4_ColMajornatural_0 viewProjection_0;
    _MatrixStorage_float4x4_ColMajornatural_0 viewProjectionNj_0;
    _MatrixStorage_float4x4_ColMajornatural_0 prevViewProjectionNj_0;
    float3 device* vertices_0;
    float2 device* uvs_0;
    Instance_natural_0 device* instances_0;
};

struct PixelData_0
{
    uint srcTexture_0;
};


#line 21 "samples/graphics/Pixel.slang"
struct EntryPointParams_0
{
    VertexData_natural_0 device* _0;
    PixelData_0 device* data_1;
};


#line 21
struct KernelContext_0
{
    texture2d<float, access::sample>  textureHeap_0[];
    sampler  ngapiSamplerHeap_0[];
    EntryPointParams_0 constant* entryPointParams_0;
};


#line 104 "ngapi/include/Sampler.h"
float4 HwSampler_Sample_0(const HwSampler_0 thread* this_0, texture2d<float, access::sample> tex_0, float2 uv_0, KernelContext_0 thread* kernelContext_0)
{
    return ((tex_0).sample((kernelContext_0->ngapiSamplerHeap_0[this_0->slot_0]), (uv_0)));
}


#line 15 "samples/graphics/Pixel.slang"
struct PixelOut_0
{
    float4 color_0 [[color(0)]];
    float4 motionVectors_0 [[color(1)]];
};


#line 15
struct pixelInput_0
{
    float2 motionVectors_1 [[user(_SLANG_ATTR)]];
    float2 uv_1 [[user(_SLANG_ATTR_1)]];
};

[[fragment]] PixelOut_0 main_0(pixelInput_0 _S2 [[stage_in]], float4 position_0 [[position]], texture2d<float, access::sample>  textureHeap_1[], sampler  ngapiSamplerHeap_1[], EntryPointParams_0 constant* entryPointParams_1 [[buffer(0)]])
{

#line 21
    thread KernelContext_0 kernelContext_1;

#line 21
    (&kernelContext_1)->textureHeap_0 = textureHeap_1;

#line 21
    (&kernelContext_1)->ngapiSamplerHeap_0 = ngapiSamplerHeap_1;

#line 21
    (&kernelContext_1)->entryPointParams_0 = entryPointParams_1;

    thread PixelOut_0 outPixel_0;
    uint _S3 = entryPointParams_1->data_1->srcTexture_0;

#line 24
    thread HwSampler_0 _S4 = aniso16Wrap_0();

#line 24
    float4 _S5 = HwSampler_Sample_0(&_S4, textureHeap_1[_S3], _S2.uv_1, &kernelContext_1);

#line 24
    (&outPixel_0)->color_0 = _S5;
    (&outPixel_0)->motionVectors_0 = float4(_S2.motionVectors_1, 0.0, 0.0);
    return outPixel_0;
}

