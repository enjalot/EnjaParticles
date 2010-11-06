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
           __global float4* vars_unsorted
           //__global uint* sort_indexes
		   )
{
    // particle index
    uint i = get_global_id(0);
	// do not use gp->numParticles (since it numParticles changed via define)
	//int num = get_global_size(0);
    if (i >= num) return;  // num: 512

    //index = sort_indexes[i];

    unsorted_density(i) = density[i];
    unsorted_pos(i) = position[i];
    unsorted_vel(i) = velocity[i];
    unsorted_veleval(i) = veleval[i];

}


#endif
