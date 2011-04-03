#include "cl_macros.h"
#include "cl_structs.h"

float magnitude(float4 vec)
{
    return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}       

__kernel void euler(
                   //__global float4* vars_unsorted, 
                   //__global float4* vars_sorted, 
                   //__global float4* positions,  // for VBO 
                   __global float4* pos_u, 
                   __global float4* pos_s, 
                   __global float4* vel_u, 
                   __global float4* vel_s, 
                   __global float4* force_s, 
                   __global int* sort_indices,  
                   //		__global float4* color,
                   __constant struct SPHParams* sphp, 
                   float dt)
{
    unsigned int i = get_global_id(0);
    int num = sphp->num;
    if (i >= num) return;



    float4 p = pos_s[i];
    float4 v = vel_s[i];
    float4 f = force_s[i];

    //external force is gravity
    f.z += -9.8f;

    float speed = magnitude(f);
    if (speed > 600.0f) //velocity limit, need to pass in as struct
    {
        f *= 600.0f/speed;
    }

    v += dt*f;
    //p += dt*v / sphp->simulation_scale;
    p += dt*v;
    p.w = 1.0f; //just in case
    p.xyz /= sphp->simulation_scale;

    uint originalIndex = sort_indices[i];

    vel_u[originalIndex] = v;
    //unsorted_veleval(originalIndex) = v;
    //float dens = density(i);
    //unsorted_pos(originalIndex) = (float4)(p.xyz, dens);
    pos_u[originalIndex] = (float4)(p.xyz, 1.);  // for plotting

}
