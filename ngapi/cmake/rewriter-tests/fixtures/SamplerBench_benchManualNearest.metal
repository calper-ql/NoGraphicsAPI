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


#line 104 "samples/samplerbench/SamplerBench.slang"
struct EntryPointParams_0
{
    SamplerBenchData_0 device* data_0;
};


#line 104
struct KernelContext_0
{
    texture2d<float, access::sample>  textureHeap_0[];
    texture2d<float, access::read_write>  rwTextureHeap_0[];
    EntryPointParams_0 constant* entryPointParams_0;
};


#line 104
[[kernel]] void benchManualNearest(uint3 threadId_0 [[thread_position_in_grid]], texture2d<float, access::sample>  textureHeap_1[], texture2d<float, access::read_write>  rwTextureHeap_1[], EntryPointParams_0 constant* entryPointParams_1 [[buffer(0)]])
{

#line 104
    thread KernelContext_0 kernelContext_0;

#line 104
    (&kernelContext_0)->textureHeap_0 = textureHeap_1;

#line 104
    (&kernelContext_0)->rwTextureHeap_0 = rwTextureHeap_1;

#line 104
    (&kernelContext_0)->entryPointParams_0 = entryPointParams_1;

    int _S1 = int(entryPointParams_1->data_0->imageSize_0.x);

#line 106
    int _S2 = int(entryPointParams_1->data_0->imageSize_0.y);

#line 106
    int2 imgSize_0 = int2(_S1, _S2);
    int _S3 = int(threadId_0.x);

#line 107
    int _S4 = int(threadId_0.y);

#line 107
    int2 pixelCoord_1 = int2(_S3, _S4);

#line 107
    bool _S5;
    if(_S3 >= _S1)
    {

#line 108
        _S5 = true;

#line 108
    }
    else
    {

#line 108
        _S5 = _S4 >= _S2;

#line 108
    }

#line 108
    if(_S5)
    {

#line 109
        return;
    }
    uint _S6 = entryPointParams_1->data_0->srcTexture_0;

#line 111
    texture2d<float, access::sample>  _S7[] = (&kernelContext_0)->textureHeap_0;
    float2 _S8 = float2(imgSize_0);

#line 112
    float2 _S9 = float2(1.0)  / _S8;

    int r_0 = entryPointParams_1->data_0->kernelRadius_0;
    float4 _S10 = float4(0.0, 0.0, 0.0, 0.0);
    int _S11 = - entryPointParams_1->data_0->kernelRadius_0;

#line 116
    int ky_1 = _S11;

#line 116
    float4 acc_0 = _S10;

#line 116
    for(;;)
    {

#line 116
        if(ky_1 <= r_0)
        {
        }
        else
        {

#line 116
            break;
        }

#line 116
        int kx_1 = _S11;

        for(;;)
        {

#line 118
            if(kx_1 <= r_0)
            {
            }
            else
            {

#line 118
                break;
            }


            int3 _S12 = int3(clamp(int2(floor(tapUV_0(pixelCoord_1, kx_1, ky_1, _S9) * _S8)), int2(int(0), int(0)), imgSize_0 - int2(int(1), int(1))), int(0));

#line 122
            float4 acc_1 = acc_0 + ((_S7[_S6]).read(vec<uint,2>(((_S12)).xy), uint(((_S12)).z)));

#line 118
            kx_1 = kx_1 + int(1);

#line 118
            acc_0 = acc_1;

#line 118
        }

#line 116
        ky_1 = ky_1 + int(1);

#line 116
    }

#line 126
    int _S13 = int(2) * r_0 + int(1);
    (&kernelContext_0)->rwTextureHeap_0[entryPointParams_1->data_0->dstTexture_0].write(acc_0 / float4(float(_S13 * _S13)) ,uint2(pixelCoord_1));
    return;
}

