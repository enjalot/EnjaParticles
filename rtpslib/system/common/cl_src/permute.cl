// This software contains source code provided by NVIDIA Corporation.
// Specifically code from the CUDA 2.3 SDK "Particles" sample

#ifndef _PERMUTE_
#define _PERMUTE_

#include "cl_macros.h"
#include "cl_structs.h"

//#pragma cl_khr_global_int32_base_atomics : enable
//----------------------------------------------------------------------
__kernel void permute(
                            int num,
                            __global float4* pos_u,
                            __global float4* pos_s,
                            __global float4* vel_u,
                            __global float4* vel_s,
                            __global float4* veleval_u,
                            __global float4* veleval_s,
                            __global float4*   color_u,
                            __global float4*   color_s,
                            __global uint* sort_indices
                            )
{
    uint index = get_global_id(0);
    //int num = sphp->num;
   if (index >= num) return;
    //cell_indices_end[index] = 42;
    uint sorted_index = sort_indices[index];
    pos_s[index]     = pos_u[sorted_index];// * sphp->simulation_scale;
    vel_s[index]     = vel_u[sorted_index];
    veleval_s[index] = veleval_u[sorted_index]; // not sure if needed
    color_s[index]   = color_u[sorted_index];
    //density(index) = unsorted_density(sorted_index); // only for debugging
}
//----------------------------------------------------------------------

#endif
