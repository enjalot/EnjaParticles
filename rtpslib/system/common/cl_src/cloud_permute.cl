// This software contains source code provided by NVIDIA Corporation.
// Specifically code from the CUDA 2.3 SDK "Particles" sample

#ifndef _CLOUDPERMUTE_
#define _CLOUDPERMUTE_

#include "cl_macros.h"
#include "cl_structs.h"

//#pragma cl_khr_global_int32_base_atomics : enable
//----------------------------------------------------------------------
__kernel void cloud_permute(
                            int num,
                            __global float4* pos_u,
                            __global float4* pos_s,
                            __global float4* normal_u,
                            __global float4* normal_s,
                            __global uint* sort_indices
                            )
{
    uint index = get_global_id(0);

    if (index >= num) return;
    uint sorted_index = sort_indices[index];

	// Clouds are now in simulation coordinates
    pos_s[index]     = pos_u[sorted_index]; 
    normal_s[index]     = normal_u[sorted_index];
}
//----------------------------------------------------------------------

#endif
