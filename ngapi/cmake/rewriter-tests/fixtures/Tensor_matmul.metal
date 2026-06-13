#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 23 "samples/learning/Common.h"
struct TensorMatMulData_0
{
    ulong n_0;
    ulong a_0;
    ulong b_0;
    ulong c_0;
    float device* x_0;
    float device* y_0;
    float device* z_0;
};


#line 70 "samples/learning/Tensor.slang"
struct EntryPointParams_0
{
    TensorMatMulData_0 device* data_0;
};


#line 70
[[kernel]] void _matmul(uint3 t_0 [[thread_position_in_grid]], EntryPointParams_0 constant* entryPointParams_0 [[buffer(0)]])
{

#line 70
    uint t_1 = t_0.x;

    ulong _S1 = ulong(t_1);

#line 72
    if(_S1 >= (entryPointParams_0->data_0->n_0))
    {
        return;
    }

    ulong _S2 = _S1 / entryPointParams_0->data_0->c_0;
    ulong _S3 = _S1 % entryPointParams_0->data_0->c_0;

#line 78
    int i_0 = int(0);

#line 78
    float d_0 = 0.0;


    for(;;)
    {

#line 81
        ulong _S4 = ulong(i_0);

#line 81
        if(_S4 < (entryPointParams_0->data_0->b_0))
        {
        }
        else
        {

#line 81
            break;
        }
        float d_1 = d_0 + *(entryPointParams_0->data_0->x_0 + (_S2 * entryPointParams_0->data_0->b_0 + _S4)) * *(entryPointParams_0->data_0->y_0 + (entryPointParams_0->data_0->c_0 * _S4 + _S3));

#line 81
        i_0 = i_0 + int(1);

#line 81
        d_0 = d_1;

#line 81
    }

#line 86
    *(entryPointParams_0->data_0->z_0 + t_1) = d_0;
    return;
}

