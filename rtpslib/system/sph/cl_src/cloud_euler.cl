#include "cl_macros.h"
#include "cl_structs.h"

//float magnitude(float4 vec)
//{
    //return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
//}       

__kernel void cloudEuler(
                   __global float4* pos_u, 
                   __global float4* pos_s, 
                   float4 vel, 
                   __global int* sort_indices,  
                   __constant struct SPHParams* sphp, 
                   float dt)
{
    unsigned int i = get_global_id(0);
    int num = sphp->num;
    if (i >= num) return;

    float4 p = pos_s[i];

    p += dt*vel;
    p.w = 1.0f; //just in case
    p.xyz /= sphp->simulation_scale;

    uint originalIndex = sort_indices[i];

    pos_u[originalIndex] = (float4)(p.xyz, 1.);  // for plotting
}

