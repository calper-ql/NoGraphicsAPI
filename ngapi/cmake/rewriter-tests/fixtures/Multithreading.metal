#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 1 "token paste"
constant ulong fc_linearClamp_ngapiSlot_0 [[function_constant(0)]];
constant ulong linearClamp_ngapiSlot_0 = is_function_constant_defined(fc_linearClamp_ngapiSlot_0) ? fc_linearClamp_ngapiSlot_0 : 5640617247111123531ULL;

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


#line 6 "samples/multithreading/Multithreading.slang"
HwSampler_0 linearClamp_0()
{

#line 137 "ngapi/include/Sampler.h"
    return HwSampler_x24init_0(uint(linearClamp_ngapiSlot_0));
}


#line 6 "samples/multithreading/Multithreading.h"
struct WorkerData_0
{
    uint count_0;
    uint workerId_0;
    uint srcTexture_0;
    float device* input_0;
    float device* output_0;
};


#line 8 "samples/multithreading/Multithreading.slang"
struct EntryPointParams_0
{
    WorkerData_0 device* data_0;
};


#line 8
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


#line 8 "samples/multithreading/Multithreading.slang"
[[kernel]] void main_0(uint3 threadId_0 [[thread_position_in_grid]], texture2d<float, access::sample>  textureHeap_1[], sampler  ngapiSamplerHeap_1[], EntryPointParams_0 constant* entryPointParams_1 [[buffer(0)]])
{

#line 8
    thread KernelContext_0 kernelContext_1;

#line 8
    (&kernelContext_1)->textureHeap_0 = textureHeap_1;

#line 8
    (&kernelContext_1)->ngapiSamplerHeap_0 = ngapiSamplerHeap_1;

#line 8
    (&kernelContext_1)->entryPointParams_0 = entryPointParams_1;

    uint _S2 = threadId_0.x;

#line 10
    if(_S2 >= (entryPointParams_1->data_0->count_0))
    {

#line 11
        return;
    }
    uint _S3 = entryPointParams_1->data_0->srcTexture_0;

#line 13
    float2 _S4 = float2(0.5, 0.5);

#line 13
    thread HwSampler_0 _S5 = linearClamp_0();

#line 13
    float4 _S6 = HwSampler_SampleLevel_0(&_S5, (&kernelContext_1)->textureHeap_0[_S3], _S4, 0.0, &kernelContext_1);
    *(entryPointParams_1->data_0->output_0 + _S2) = *(entryPointParams_1->data_0->input_0 + _S2) + float(entryPointParams_1->data_0->workerId_0) * _S6.x;
    return;
}

