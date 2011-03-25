#ifndef _PREP_CL_H_
#define _PREP_CL_H_



#include "cl_structs.h"
#include "cl_macros.h"

__kernel void prep(
                  int stage,
                  __global float4* position,
                  __global float4* velocity,
                  __global float4* vars_unsorted,
                  __global float4* vars_sorted,
                  __global uint* sort_indices,
                  __constant struct SPHParams* sphp
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
    int num = sphp->num;
    if (i >= num) return;

    uint index = sort_indices[i];

    if (stage == 1)
    {
        unsorted_pos(i) = position[i];
        unsorted_vel(i) = velocity[i];

    }
    else if (stage == 0)
    {
        velocity[index] = vel(i);
    }
    if (stage == 2)
    {
        //used if we need to copy sorted positions into positions array
        //later we also need to copy color

        position[i] = pos(i);
        velocity[i] = vel(i);
        unsorted_pos(i) = pos(i);
        //unsorted_pos(i) = (float4)(1166., 66., 66., 66.);
    }
    /*
    unsorted_density(i) = density[i];
    unsorted_vel(i) = velocity[i];
    unsorted_veleval(i) = veleval[i];
    unsorted_force(i) = force[i];
    */


}


#endif
