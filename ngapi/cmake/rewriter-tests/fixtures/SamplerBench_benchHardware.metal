#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

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


#line 18 "samples/samplerbench/SamplerBench.slang"
struct EntryPointParams_0
{
    SamplerBenchData_0 device* data_0;
};


#line 18
struct KernelContext_0
{
    texture2d<float, access::sample>  textureHeap_0[];
    sampler  ngapiSamplerHeap_0[];
    texture2d<float, access::read_write>  rwTextureHeap_0[];
    EntryPointParams_0 constant* entryPointParams_0;
};


#line 18
[[kernel]] void benchHardware(uint3 threadId_0 [[thread_position_in_grid]], texture2d<float, access::sample>  textureHeap_1[], sampler  ngapiSamplerHeap_1[], texture2d<float, access::read_write>  rwTextureHeap_1[], EntryPointParams_0 constant* entryPointParams_1 [[buffer(0)]])
{

#line 18
    thread KernelContext_0 kernelContext_0;

#line 18
    (&kernelContext_0)->textureHeap_0 = textureHeap_1;

#line 18
    (&kernelContext_0)->ngapiSamplerHeap_0 = ngapiSamplerHeap_1;

#line 18
    (&kernelContext_0)->rwTextureHeap_0 = rwTextureHeap_1;

#line 18
    (&kernelContext_0)->entryPointParams_0 = entryPointParams_1;

    int _S1 = int(entryPointParams_1->data_0->imageSize_0.x);

#line 20
    int _S2 = int(entryPointParams_1->data_0->imageSize_0.y);

#line 20
    int2 imgSize_0 = int2(_S1, _S2);
    int _S3 = int(threadId_0.x);

#line 21
    int _S4 = int(threadId_0.y);

#line 21
    int2 pixelCoord_1 = int2(_S3, _S4);

#line 21
    bool _S5;
    if(_S3 >= _S1)
    {

#line 22
        _S5 = true;

#line 22
    }
    else
    {

#line 22
        _S5 = _S4 >= _S2;

#line 22
    }

#line 22
    if(_S5)
    {

#line 23
        return;
    }
    uint _S6 = entryPointParams_1->data_0->srcTexture_0;

#line 25
    texture2d<float, access::sample>  _S7[] = (&kernelContext_0)->textureHeap_0;

#line 25
    sampler  _S8[] = (&kernelContext_0)->ngapiSamplerHeap_0;

    float2 _S9 = float2(1.0)  / float2(imgSize_0);

    int r_0 = entryPointParams_1->data_0->kernelRadius_0;
    float4 _S10 = float4(0.0, 0.0, 0.0, 0.0);
    int _S11 = - entryPointParams_1->data_0->kernelRadius_0;

#line 31
    int ky_1 = _S11;

#line 31
    float4 acc_0 = _S10;

#line 31
    for(;;)
    {

#line 31
        if(ky_1 <= r_0)
        {
        }
        else
        {

#line 31
            break;
        }

#line 31
        int kx_1 = _S11;
        for(;;)
        {

#line 32
            if(kx_1 <= r_0)
            {
            }
            else
            {

#line 32
                break;
            }

#line 33
            float4 acc_1 = acc_0 + ((_S7[_S6]).sample((_S8[int(0)]), (tapUV_0(pixelCoord_1, kx_1, ky_1, _S9)), level((0.0))));

#line 32
            kx_1 = kx_1 + int(1);

#line 32
            acc_0 = acc_1;

#line 32
        }

#line 31
        ky_1 = ky_1 + int(1);

#line 31
    }



    int _S12 = int(2) * r_0 + int(1);
    (&kernelContext_0)->rwTextureHeap_0[entryPointParams_1->data_0->dstTexture_0].write(acc_0 / float4(float(_S12 * _S12)) ,uint2(pixelCoord_1));
    return;
}

