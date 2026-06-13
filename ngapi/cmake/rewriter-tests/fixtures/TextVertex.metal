#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 21 "samples/common/TextVertex.slang"
array<float2, int(4)> _S1()
{

#line 10
    array<float2, int(4)> _S2 = { float2(0.0, 0.0), float2(1.0, 0.0), float2(1.0, 1.0), float2(0.0, 1.0) };

#line 10
    return _S2;
}


#line 3
struct VertexOut_0
{
    float4 position_0;
    float2 uv_0;
};


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


#line 17 "samples/common/TextVertex.slang"
struct EntryPointParams_0
{
    TextVertexData_0 device* data_0;
    TextPixelData_0 device* _0;
};


#line 17
struct KernelContext_0
{
    EntryPointParams_0 constant* entryPointParams_0;
    array<float2, int(4)> quadVertices_0;
};


#line 17
VertexOut_0 main_0(const uint thread* vertexId_0, const uint thread* instanceId_0, KernelContext_0 thread* kernelContext_0)
{


    float2 quadPos_0 = kernelContext_0->quadVertices_0[*vertexId_0];

#line 19
    thread VertexOut_0 outVertex_0;

#line 25
    (&outVertex_0)->position_0 = float4((float2(float(*instanceId_0 * kernelContext_0->entryPointParams_0->data_0->textWidth_0), 0.0) + kernelContext_0->quadVertices_0[*vertexId_0] * float2(float(kernelContext_0->entryPointParams_0->data_0->textWidth_0), float(kernelContext_0->entryPointParams_0->data_0->textHeight_0))) / float2(float(kernelContext_0->entryPointParams_0->data_0->width_0), float(kernelContext_0->entryPointParams_0->data_0->height_0)) * float2(2.0)  - float2(1.0) , 0.0, 1.0);


    uint charCode_0 = uint(*(kernelContext_0->entryPointParams_0->data_0->text_0 + *instanceId_0));
    uint charsPerRow_0 = kernelContext_0->entryPointParams_0->data_0->atlasWidth_0 / kernelContext_0->entryPointParams_0->data_0->textWidth_0;
    uint charX_0 = charCode_0 % charsPerRow_0;
    uint charY_0 = charCode_0 / charsPerRow_0;
    (&outVertex_0)->uv_0 = (float2(float(charX_0), float(charY_0)) + quadPos_0) * float2(float(kernelContext_0->entryPointParams_0->data_0->textWidth_0), float(kernelContext_0->entryPointParams_0->data_0->textHeight_0)) / float2(float(kernelContext_0->entryPointParams_0->data_0->atlasWidth_0), float(kernelContext_0->entryPointParams_0->data_0->atlasHeight_0));

    return outVertex_0;
}


#line 34
struct main_Result_0
{
    float4 position_1 [[position]];
    float2 uv_1 [[user(_SLANG_ATTR)]];
};


#line 34
[[vertex]] main_Result_0 main_1(uint vertexId_1 [[vertex_id]], uint instanceId_1 [[instance_id]])
{

#line 34
    thread KernelContext_0 kernelContext_1;

#line 34
    (&kernelContext_1)->quadVertices_0 = _S1();

#line 34
    thread uint _S3 = instanceId_1;

#line 34
    thread uint _S4 = vertexId_1;

#line 34
    VertexOut_0 _S5 = main_0(&_S4, &_S3, &kernelContext_1);

#line 34
    thread main_Result_0 _S6;

#line 34
    (&_S6)->position_1 = _S5.position_0;

#line 34
    (&_S6)->uv_1 = _S5.uv_0;

#line 34
    return _S6;
}

