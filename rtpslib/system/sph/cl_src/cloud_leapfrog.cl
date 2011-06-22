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


