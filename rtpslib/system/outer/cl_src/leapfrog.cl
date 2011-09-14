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



#include "cl_macros.h"
#include "cl_structs.h"


__kernel void leapfrog(
                      //__global float4* vars_unsorted, 
                      //__global float4* vars_sorted, 
                      __global float4* pos_u,
                      __global float4* pos_s,
                      __global float4* vel_u,
                      __global float4* vel_s,
                      __global float4* veleval_u,
                      __global float4* force_s,
                      __global float4* xsph_s,
                      __global int* sort_indices,  
                      //		__global float4* color,
                      __constant struct SPHParams* sphp, 
                      float dt)
{
    unsigned int i = get_global_id(0);
    //int num = get_global_size(0); // for access functions in cl_macros.h
    int num = sphp->num;
    if (i >= num) return;

    /*
    float4 p = pos(i);
    float4 v = vel(i);
    float4 f = force(i);
    */

    float4 p = pos_s[i] * sphp->simulation_scale;
    float4 v = vel_s[i];
    float4 f = force_s[i];




    //external force is gravity
    f.z += sphp->gravity;
    f.w = 0.f;

    float speed = length(f);
    if (speed > sphp->velocity_limit) //velocity limit, need to pass in as struct
    {
        f *= sphp->velocity_limit/speed;
    }

    float4 vnext = v + dt*f;
    //float4 vnext = v;// + dt*f;
    vnext += sphp->xsph_factor * xsph_s[i];

    float4 veval = 0.5f*(v+vnext);

#if 0
    //crazy velocity freezing effect
    float x = p.x / sphp->simulation_scale;
    if (x > 4.)
    {
        float mv = length( (float4)(vnext.xyz, 0.0f));
        //vnext /= mv;
        //vnext *= log(mv); 
        //this should be changed to decay with lifetime
        veval = (float4)(0.0, 0.0, 0.0, 0.0);
        vnext = (float4)(0.0, 0.0, 0.0, 0.0);
    }
#endif


    p += dt * vnext;
    p.w = 1.0f; //just in case

    //Not sure why we put them back in unsorted order
    //might as well write them back in order and save some memory access costs
    //uint originalIndex = sort_indices[i];
    //uint originalIndex = i;

    //float dens = density(i);
    p.xyz /= sphp->simulation_scale;


    //unsorted_pos(originalIndex) = (float4)(pos(i).xyz / sphp->simulation_scale, 1.);
    //unsorted_pos(originalIndex)     = (float4)(p.xyz, dens);
    //unsorted_pos(originalIndex)     = p;
    ///unsorted_vel(originalIndex)     = vnext;
    ///unsorted_veleval(originalIndex) = veval; 
    ///positions[originalIndex]        = (float4)(p.xyz, 1.0f);  // for plotting
    
    vel_u[i] = vnext;
    veleval_u[i] = veval; 
    pos_u[i] = (float4)(p.xyz, 1.0f);  // for plotting
    
    
    
    //	color[originalIndex]			= surface(i);
    //positions[originalIndex] = unsorted_pos(originalIndex);
    //positions[i] = unsorted_pos(i);

    // FOR DEBUGGING
    //unsorted_force(originalIndex) 	= f; // FOR DEBUGGING ONLY
    //unsorted_density(originalIndex) = dens; // FOR DEBUGGING ONLY
    //positions[originalIndex] 		= (float4)(p.xyz, dens);  // for plotting
}


