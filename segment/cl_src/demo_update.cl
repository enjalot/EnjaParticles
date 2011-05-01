
#include "cl_macros.h"
#include "cl_structs.h"


__kernel void update(
                      __global float4* vel_u,
                      __global float4* col_u,
                      __global float4* force_s,
                      __global float* density_s,
                      __global int* sort_indices,  
                      __constant struct SPHParams* sphp, 
                      float dt
                      )
{
    unsigned int i = get_global_id(0);
    //int num = get_global_size(0); // for access functions in cl_macros.h
    int num = sphp->num;
    if (i >= num) return;

    //int index = sort_indices[i];
    float4 f = force_s[i];
    float d = density_s[i];

    float speed = length(f);
    float fnorm = speed / sphp->velocity_limit * 10.0f;

    float dnorm = d / 2000.f;
    dnorm = 1.4f - clamp(dnorm, 0.0f, 1.0f);

    //col_u[i].xyz = f.xyz;
    col_u[i].x = fnorm;
    col_u[i].y = d;
    col_u[i].z = 0.f;
    col_u[i].w = dnorm;
   
}


