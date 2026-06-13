#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 6 "samples/learning/Common.h"
struct TensorData_0
{
    ulong n_0;
    float device* x_0;
    float device* y_0;
    float device* z_0;
};


#line 89 "samples/learning/Tensor.slang"
struct EntryPointParams_0
{
    TensorData_0 device* data_0;
};


#line 89
[[kernel]] void _pow(uint3 t_0 [[thread_position_in_grid]], EntryPointParams_0 constant* entryPointParams_0 [[buffer(0)]])
{

#line 89
    uint t_1 = t_0.x;

    if(ulong(t_1) >= (entryPointParams_0->data_0->n_0))
    {
        return;
    }
    *(entryPointParams_0->data_0->z_0 + t_1) = pow(*(entryPointParams_0->data_0->x_0 + t_1), *(entryPointParams_0->data_0->y_0 + t_1));
    return;
}

