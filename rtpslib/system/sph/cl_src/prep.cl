/****************************************************************************************
* Real-Time Particle System - An OpenCL based Particle system developed to run on modern GPUs. Includes SPH fluid simulations.
* version 1.0, September 14th 2011
* 
* Copyright (C) 2011 Ian Johnson, Andrew Young, Gordon Erlebacher, Myrna Merced, Evan Bollig
* 
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* 
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
****************************************************************************************/


#ifndef _PREP_CL_H_
#define _PREP_CL_H_



#include "cl_structs.h"
#include "cl_macros.h"

__kernel void prep(
                  int stage,
                  __global float4* pos_u,
                  __global float4* pos_s,
                  __global float4* vel_u,
                  __global float4* vel_s,
                  __global float4* veleval_u,
                  __global float4* veleval_s,
                  //__global float4* vars_unsorted,
                  //__global float4* vars_sorted,
                  __global float4* color_u,
                  __global float4* color_s,
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
    if (stage == 0)
    {
        //velocity[index] = vel(i);
        //vel_u[index] = vel_s[i];
        //veleval_u[index] = veleval_s[i];
        //color_u[index] = color_s[i];
    }
    else if (stage == 1)
    {
        //unsorted_pos(i) = position[i];
        //unsorted_vel(i) = velocity[i];

        //pos_u[i] = pos_s[i];
        //vel_u[i] = vel_s[i];
        //veleval_u[i] = veleval_s[i];
        //color_u[i] = color_s[i];

    }
    if (stage == 2)
    {
        //used if we need to copy sorted positions into positions array
        //later we also need to copy color

        /*
        position[i] = pos(i);
        velocity[i] = vel(i);
        unsorted_pos(i) = pos(i);
        */

        pos_u[i] = pos_s[i];
        vel_u[i] = vel_s[i];
        veleval_u[i] = veleval_s[i];
        //unsorted_pos(i) = pos(i);
        //color_u[i] = color_s[i];
    }
    /*
    unsorted_density(i) = density[i];
    unsorted_vel(i) = velocity[i];
    unsorted_veleval(i) = veleval[i];
    unsorted_force(i) = force[i];
    */


}


#endif
