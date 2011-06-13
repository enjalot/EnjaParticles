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
                            //__global float4* veleval_u,
                            //__global float4* veleval_s,
                            //__global float4*   color_u,
                            //__global float4*   color_s,
                            __global uint* sort_indices
                            )
{
    uint index = get_global_id(0);
    //int num = sphp->num;
   if (index >= num) return;
    //cell_indices_end[index] = 42;
    uint sorted_index = sort_indices[index];
    pos_s[index]     = pos_u[sorted_index];// * sphp->simulation_scale;
    normal_s[index]     = normal_u[sorted_index];
    //veleval_s[index] = veleval_u[sorted_index]; // not sure if needed
    //color_s[index]   = color_u[sorted_index];
    //density(index) = unsorted_density(sorted_index); // only for debugging
}
//----------------------------------------------------------------------

#endif
