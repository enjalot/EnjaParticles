
#include "cl_macros.h"
#include "cl_structs.h"


__kernel void leapfrog(
                      __global int* sort_indices,  
                      __global float4* vars_unsorted, 
                      __global float4* vars_sorted, 
                      __global float4* positions,  // for VBO 
                      //		__global float4* color,
                      __constant struct SPHParams* sphp, 
                      float dt)
{
    unsigned int i = get_global_id(0);
    //int num = get_global_size(0); // for access functions in cl_macros.h
    int num = sphp->num;
    if (i >= num) return;

    float4 p = pos(i);
    float4 v = vel(i);
    float4 f = force(i);

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
    // WHY IS MY CORRECTION NEGATIVE and IAN's POSITIVE? 
    vnext += sphp->xsph_factor * xsph(i);

    p += dt * vnext;
    p.w = 1.0f; //just in case
    float4 veval = 0.5f*(v+vnext);

    //Not sure why we put them back in unsorted order
    //might as well write them back in order and save some memory access costs
    uint originalIndex = sort_indices[i];
    //this does mess some things up right now, with very slight speed increase
    //uint originalIndex = i;

    // writeback to unsorted buffer
    float dens = density(i);
    p.xyz /= sphp->simulation_scale;


    unsorted_pos(originalIndex)     = (float4)(p.xyz, dens);
    //unsorted_pos(originalIndex) = (float4)(pos(i).xyz / sphp->simulation_scale, 1.);
    unsorted_vel(originalIndex)     = vnext;
    unsorted_veleval(originalIndex) = veval; 
    positions[originalIndex]        = (float4)(p.xyz, 1.);  // for plotting
    //	color[originalIndex]			= surface(i);
    //positions[originalIndex] = unsorted_pos(originalIndex);
    //positions[i] = unsorted_pos(i);

    // FOR DEBUGGING
    //unsorted_force(originalIndex) 	= f; // FOR DEBUGGING ONLY
    //unsorted_density(originalIndex) = dens; // FOR DEBUGGING ONLY
    //positions[originalIndex] 		= (float4)(p.xyz, dens);  // for plotting
}


