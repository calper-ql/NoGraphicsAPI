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


#line 39 "samples/learning/Tensor.slang"
struct EntryPointParams_0
{
    TensorData_0 device* data_0;
};


#line 39
[[kernel]] void _dot(uint3 t_0 [[thread_position_in_grid]], EntryPointParams_0 constant* entryPointParams_0 [[buffer(0)]])
{
    if(ulong(t_0.x) >= (entryPointParams_0->data_0->n_0))
    {
        return;
    }

#line 43
    int i_0 = int(0);

#line 43
    float result_0 = 0.0;



    for(;;)
    {

#line 47
        if(ulong(i_0) < (entryPointParams_0->data_0->n_0))
        {
        }
        else
        {

#line 47
            break;
        }
        float result_1 = result_0 + *(entryPointParams_0->data_0->x_0 + i_0) * *(entryPointParams_0->data_0->y_0 + i_0);

#line 47
        i_0 = i_0 + int(1);

#line 47
        result_0 = result_1;

#line 47
    }



    *(entryPointParams_0->data_0->z_0 + int(0)) = result_0;
    return;
}

