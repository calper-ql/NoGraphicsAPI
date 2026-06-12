#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 3 "samples/graphics/Vertex.slang"
struct VertexOut_0
{
    float4 position_0;
    float2 motionVectors_0;
    float2 uv_0;
};


#line 14 "samples/graphics/Graphics.h"
struct _MatrixStorage_float4x4_ColMajornatural_0
{
    array<float4, int(4)> data_0;
};


#line 14
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


#line 10 "samples/graphics/Vertex.slang"
struct EntryPointParams_0
{
    VertexData_natural_0 device* data_1;
    PixelData_0 device* _0;
};


#line 10
struct KernelContext_0
{
    EntryPointParams_0 constant* entryPointParams_0;
};


#line 10
VertexOut_0 main_0(const uint thread* vertexId_0, const uint thread* instanceId_0, KernelContext_0 thread* kernelContext_0)
{

    float4 position_1 = float4(*(kernelContext_0->entryPointParams_0->data_1->vertices_0 + *vertexId_0), 1.0);

#line 12
    thread VertexOut_0 outVertex_0;

    (&outVertex_0)->position_0 = ((((((position_1) * (matrix<float,int(4),int(4)> ((kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(0)][int(0)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(1)][int(0)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(2)][int(0)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(3)][int(0)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(0)][int(1)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(1)][int(1)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(2)][int(1)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(3)][int(1)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(0)][int(2)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(1)][int(2)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(2)][int(2)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(3)][int(2)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(0)][int(3)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(1)][int(3)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(2)][int(3)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(3)][int(3)]))))) * (matrix<float,int(4),int(4)> (kernelContext_0->entryPointParams_0->data_1->viewProjection_0.data_0[int(0)][int(0)], kernelContext_0->entryPointParams_0->data_1->viewProjection_0.data_0[int(1)][int(0)], kernelContext_0->entryPointParams_0->data_1->viewProjection_0.data_0[int(2)][int(0)], kernelContext_0->entryPointParams_0->data_1->viewProjection_0.data_0[int(3)][int(0)], kernelContext_0->entryPointParams_0->data_1->viewProjection_0.data_0[int(0)][int(1)], kernelContext_0->entryPointParams_0->data_1->viewProjection_0.data_0[int(1)][int(1)], kernelContext_0->entryPointParams_0->data_1->viewProjection_0.data_0[int(2)][int(1)], kernelContext_0->entryPointParams_0->data_1->viewProjection_0.data_0[int(3)][int(1)], kernelContext_0->entryPointParams_0->data_1->viewProjection_0.data_0[int(0)][int(2)], kernelContext_0->entryPointParams_0->data_1->viewProjection_0.data_0[int(1)][int(2)], kernelContext_0->entryPointParams_0->data_1->viewProjection_0.data_0[int(2)][int(2)], kernelContext_0->entryPointParams_0->data_1->viewProjection_0.data_0[int(3)][int(2)], kernelContext_0->entryPointParams_0->data_1->viewProjection_0.data_0[int(0)][int(3)], kernelContext_0->entryPointParams_0->data_1->viewProjection_0.data_0[int(1)][int(3)], kernelContext_0->entryPointParams_0->data_1->viewProjection_0.data_0[int(2)][int(3)], kernelContext_0->entryPointParams_0->data_1->viewProjection_0.data_0[int(3)][int(3)]))));

    float4 positionNj_0 = ((((((position_1) * (matrix<float,int(4),int(4)> ((kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(0)][int(0)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(1)][int(0)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(2)][int(0)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(3)][int(0)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(0)][int(1)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(1)][int(1)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(2)][int(1)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(3)][int(1)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(0)][int(2)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(1)][int(2)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(2)][int(2)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(3)][int(2)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(0)][int(3)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(1)][int(3)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(2)][int(3)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->model_0.data_0[int(3)][int(3)]))))) * (matrix<float,int(4),int(4)> (kernelContext_0->entryPointParams_0->data_1->viewProjectionNj_0.data_0[int(0)][int(0)], kernelContext_0->entryPointParams_0->data_1->viewProjectionNj_0.data_0[int(1)][int(0)], kernelContext_0->entryPointParams_0->data_1->viewProjectionNj_0.data_0[int(2)][int(0)], kernelContext_0->entryPointParams_0->data_1->viewProjectionNj_0.data_0[int(3)][int(0)], kernelContext_0->entryPointParams_0->data_1->viewProjectionNj_0.data_0[int(0)][int(1)], kernelContext_0->entryPointParams_0->data_1->viewProjectionNj_0.data_0[int(1)][int(1)], kernelContext_0->entryPointParams_0->data_1->viewProjectionNj_0.data_0[int(2)][int(1)], kernelContext_0->entryPointParams_0->data_1->viewProjectionNj_0.data_0[int(3)][int(1)], kernelContext_0->entryPointParams_0->data_1->viewProjectionNj_0.data_0[int(0)][int(2)], kernelContext_0->entryPointParams_0->data_1->viewProjectionNj_0.data_0[int(1)][int(2)], kernelContext_0->entryPointParams_0->data_1->viewProjectionNj_0.data_0[int(2)][int(2)], kernelContext_0->entryPointParams_0->data_1->viewProjectionNj_0.data_0[int(3)][int(2)], kernelContext_0->entryPointParams_0->data_1->viewProjectionNj_0.data_0[int(0)][int(3)], kernelContext_0->entryPointParams_0->data_1->viewProjectionNj_0.data_0[int(1)][int(3)], kernelContext_0->entryPointParams_0->data_1->viewProjectionNj_0.data_0[int(2)][int(3)], kernelContext_0->entryPointParams_0->data_1->viewProjectionNj_0.data_0[int(3)][int(3)]))));
    float4 prevPositionNj_0 = ((((((position_1) * (matrix<float,int(4),int(4)> ((kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->prevModel_0.data_0[int(0)][int(0)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->prevModel_0.data_0[int(1)][int(0)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->prevModel_0.data_0[int(2)][int(0)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->prevModel_0.data_0[int(3)][int(0)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->prevModel_0.data_0[int(0)][int(1)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->prevModel_0.data_0[int(1)][int(1)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->prevModel_0.data_0[int(2)][int(1)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->prevModel_0.data_0[int(3)][int(1)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->prevModel_0.data_0[int(0)][int(2)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->prevModel_0.data_0[int(1)][int(2)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->prevModel_0.data_0[int(2)][int(2)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->prevModel_0.data_0[int(3)][int(2)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->prevModel_0.data_0[int(0)][int(3)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->prevModel_0.data_0[int(1)][int(3)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->prevModel_0.data_0[int(2)][int(3)], (kernelContext_0->entryPointParams_0->data_1->instances_0 + *instanceId_0)->prevModel_0.data_0[int(3)][int(3)]))))) * (matrix<float,int(4),int(4)> (kernelContext_0->entryPointParams_0->data_1->prevViewProjectionNj_0.data_0[int(0)][int(0)], kernelContext_0->entryPointParams_0->data_1->prevViewProjectionNj_0.data_0[int(1)][int(0)], kernelContext_0->entryPointParams_0->data_1->prevViewProjectionNj_0.data_0[int(2)][int(0)], kernelContext_0->entryPointParams_0->data_1->prevViewProjectionNj_0.data_0[int(3)][int(0)], kernelContext_0->entryPointParams_0->data_1->prevViewProjectionNj_0.data_0[int(0)][int(1)], kernelContext_0->entryPointParams_0->data_1->prevViewProjectionNj_0.data_0[int(1)][int(1)], kernelContext_0->entryPointParams_0->data_1->prevViewProjectionNj_0.data_0[int(2)][int(1)], kernelContext_0->entryPointParams_0->data_1->prevViewProjectionNj_0.data_0[int(3)][int(1)], kernelContext_0->entryPointParams_0->data_1->prevViewProjectionNj_0.data_0[int(0)][int(2)], kernelContext_0->entryPointParams_0->data_1->prevViewProjectionNj_0.data_0[int(1)][int(2)], kernelContext_0->entryPointParams_0->data_1->prevViewProjectionNj_0.data_0[int(2)][int(2)], kernelContext_0->entryPointParams_0->data_1->prevViewProjectionNj_0.data_0[int(3)][int(2)], kernelContext_0->entryPointParams_0->data_1->prevViewProjectionNj_0.data_0[int(0)][int(3)], kernelContext_0->entryPointParams_0->data_1->prevViewProjectionNj_0.data_0[int(1)][int(3)], kernelContext_0->entryPointParams_0->data_1->prevViewProjectionNj_0.data_0[int(2)][int(3)], kernelContext_0->entryPointParams_0->data_1->prevViewProjectionNj_0.data_0[int(3)][int(3)]))));
    (&outVertex_0)->motionVectors_0 = positionNj_0.xy / float2(positionNj_0.w)  - prevPositionNj_0.xy / float2(prevPositionNj_0.w) ;

    (&outVertex_0)->uv_0 = *(kernelContext_0->entryPointParams_0->data_1->uvs_0 + *vertexId_0);

    return outVertex_0;
}


#line 22
struct main_Result_0
{
    float4 position_2 [[position]];
    float2 motionVectors_1 [[user(_SLANG_ATTR)]];
    float2 uv_1 [[user(_SLANG_ATTR_1)]];
};


#line 22
[[vertex]] main_Result_0 main_1(uint vertexId_1 [[vertex_id]], uint instanceId_1 [[instance_id]])
{

#line 22
    thread uint _S1 = instanceId_1;

#line 22
    thread uint _S2 = vertexId_1;

#line 22
    thread KernelContext_0 kernelContext_1;

#line 22
    VertexOut_0 _S3 = main_0(&_S2, &_S1, &kernelContext_1);

#line 22
    thread main_Result_0 _S4;

#line 22
    (&_S4)->position_2 = _S3.position_0;

#line 22
    (&_S4)->motionVectors_1 = _S3.motionVectors_0;

#line 22
    (&_S4)->uv_1 = _S3.uv_0;

#line 22
    return _S4;
}

