#ifndef _PREP_CL_H_
#define _PREP_CL_H_



#include "cl_structs.h"
#include "cl_macros.h"

__kernel void prep(
           int num,
           __global float* density,
           __global float4* position,
           __global float4* velocity,
           __global float4* veleval,
           __global float4* force,
           __global float4* xsph,
           __global float4* vars_unsorted
		   )
{
    uint i = get_global_id(0);
    if (i >= num) return;  // num: 512

    //index = sort_indexes[i];

    unsorted_pos(i) = position[i];
    unsorted_density(i) = density[i];
    unsorted_vel(i) = velocity[i];
    unsorted_veleval(i) = veleval[i];
    unsorted_force(i) = force[i];


}


#endif
