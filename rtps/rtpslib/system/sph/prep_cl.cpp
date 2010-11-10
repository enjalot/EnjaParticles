#ifndef _PREP_CL_H_
#define _PREP_CL_H_



#include "cl_structs.h"
#include "cl_macros.h"

__kernel void prep(
           int num,
           int stage,
           __global float4* position,
           __global float4* vars_unsorted,
           __global float4* vars_sorted,
           __global uint* sort_indices
           /*
           __global float* density,
           __global float4* velocity,
           __global float4* veleval,
           __global float4* force,
           __global float4* xsph,
           */
		   )
{
    uint i = get_global_id(0);
    if (i >= num) return;


    if(stage == 1)
    {
        unsorted_pos(i) = position[i];
    }
    else if(stage == 0)
    {
        //we only want to do this for the old num 
        uint index = sort_indices[i];
        //unsorted_density(index) = density(i);
        unsorted_vel(index) = vel(i);
        unsorted_veleval(index) = veleval(i);
        //unsorted_force(index) = force(i);
    }

    /*
    unsorted_density(i) = density[i];
    unsorted_vel(i) = velocity[i];
    unsorted_veleval(i) = veleval[i];
    unsorted_force(i) = force[i];
    */


}


#endif
