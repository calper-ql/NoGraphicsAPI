#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 14 "samples/learning/Common.h"
struct TensorTransposeData_0
{
    ulong n_0;
    ulong r_0;
    ulong c_0;
    float device* x_0;
    float device* y_0;
};


#line 54 "samples/learning/Tensor.slang"
struct EntryPointParams_0
{
    TensorTransposeData_0 device* data_0;
};


#line 54
[[kernel]] void _mT(uint3 t_0 [[thread_position_in_grid]], EntryPointParams_0 constant* entryPointParams_0 [[buffer(0)]])
{
    if(ulong(t_0.x) >= (entryPointParams_0->data_0->n_0))
    {
        return;
    }

#line 58
    int i_0 = int(0);


    for(;;)
    {

#line 61
        ulong _S1 = ulong(i_0);

#line 61
        if(_S1 < (entryPointParams_0->data_0->r_0))
        {
        }
        else
        {

#line 61
            break;
        }

#line 61
        int j_0 = int(0);

        for(;;)
        {

#line 63
            ulong _S2 = ulong(j_0);

#line 63
            if(_S2 < (entryPointParams_0->data_0->c_0))
            {
            }
            else
            {

#line 63
                break;
            }
            *(entryPointParams_0->data_0->y_0 + (_S2 * entryPointParams_0->data_0->r_0 + _S1)) = *(entryPointParams_0->data_0->x_0 + (_S1 * entryPointParams_0->data_0->c_0 + _S2));

#line 63
            j_0 = j_0 + int(1);

#line 63
        }

#line 61
        i_0 = i_0 + int(1);

#line 61
    }

#line 68
    return;
}

