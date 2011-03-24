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
    else if (stage == 2)
    {
        //used if we need to copy sorted positions into positions array
        //later we also need to copy color

        position[i] = pos(i);
        velocity[i] = vel(i);
        unsorted_pos(i) = pos(i);
    }
    else if (stage == 3)
    {
        unsorted_pos(4) = (float4)(10., 10., 10., 1.);
        unsorted_pos(5) = (float4)(10., 10., 10., 1.);
        unsorted_pos(6) = (float4)(10., 10., 10., 1.);
        unsorted_pos(7) = (float4)(10., 10., 10., 1.);
    }

    /*
    unsorted_density(i) = density[i];
    unsorted_vel(i) = velocity[i];
    unsorted_veleval(i) = veleval[i];
    unsorted_force(i) = force[i];
    */


}


#endif
