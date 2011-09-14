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


// NOT DEBUGGED. COMPILER ERRORS!!

#include "cl_macros.h"
#include "cl_structs.h"


__kernel void cloudLeapfrog(
                      __global float4* pos_u,
                      __global float4* pos_s,
                      __global float4* vel_u,
                      __global float4* vel_s,
                      __global float4* veleval_u,
                      __global float4* force_s,
                      __global int* sort_indices,  
                      //		__global float4* color,
                      __constant struct SPHParams* sphp, 
                      float dt)
{
    unsigned int i = get_global_id(0);
    //int num = get_global_size(0); // for access functions in cl_macros.h
    int num = sphp->num;
    if (i >= num) return;

    float4 p = pos_s[i] * sphp->simulation_scale;
    float4 v = vel_s[i];
    //float4 f = force_s[i];


    //external force is gravity
    //f.z += sphp->gravity;
    //f.w = 0.f;

    float speed = length(f);
    if (speed > sphp->velocity_limit) //velocity limit, need to pass in as struct
    {
        f *= sphp->velocity_limit/speed;
    }

    //float4 vnext = v + dt*f;
    //float4 veval = 0.5f*(v+vnext);

    p += dt * vnext;
    p.w = 1.0f; //just in case

    //Not sure why we put them back in unsorted order
    //might as well write them back in order and save some memory access costs
    //uint originalIndex = sort_indices[i];
    //uint originalIndex = i;

    //float dens = density(i);
    p.xyz /= sphp->simulation_scale;


    vel_u[i] = vnext;
    veleval_u[i] = veval; 
    pos_u[i] = (float4)(p.xyz, 1.0f);  // for plotting
}


// in prep
// copy from cl_position_s to cl_position_u
